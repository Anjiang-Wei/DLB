/* Copyright 2023 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cassert>
#include <map>
#include <vector>
#include <chrono>
#include <thread>
#include "mappers/default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  COMPUTE_TASK_ID,
  LAST_TASK_ID,
};

#define num_tasks (40)
#define num_cpus (4)
#define wait_ms (500)

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  TaskLauncher tl1(COMPUTE_TASK_ID, TaskArgument(NULL, 0));
  for (int i = 0; i < num_tasks; i++)
  {
    runtime->execute_task(ctx, tl1);
  }
  runtime->issue_execution_fence(ctx);
  TaskLauncher tl2(LAST_TASK_ID, TaskArgument(NULL, 0));
  runtime->execute_task(ctx, tl2);
}

void compute_task(const Task *task,
                  const std::vector<PhysicalRegion> &regions,
                  Context ctx, Runtime *runtime)
{
  std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
  return;
}

void last_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime)
{
  return;
}

class DLBMapper : public DefaultMapper
{
public:
  DLBMapper(MapperRuntime *rt, Machine machine, Processor local,
            const char *mapper_name);
  void select_task_options(const MapperContext    ctx,
                           const Task&            task,
                                 TaskOptions&     output) override;
  void select_steal_targets(const MapperContext         ctx,
                            const SelectStealingInput&  input,
                                SelectStealingOutput& output) override;
  void permit_steal_request(const MapperContext       ctx,
                            const StealRequestInput&  input,
                               StealRequestOutput& output) override;
  void default_policy_select_target_processors(MapperContext ctx,
                                               const Task &task,
                                               std::vector<Processor> &target_procs) override;
  MapperSyncModel get_mapper_sync_model() const override;
  void select_tasks_to_map(const MapperContext ctx,
                           const SelectMappingInput& input,
                               SelectMappingOutput& output) override;
   void map_task(const MapperContext ctx,
                const Task& task,
                const MapTaskInput& input,
                MapTaskOutput& output) override;
  void report_profiling(const MapperContext ctx,
                        const Task& task,
                        const TaskProfilingInfo& input) override;
private:
  // InFlightTask represents a currently executing task.
  struct InFlightTask {
    // A unique identifier for an instance of a task.
    UniqueID id;
    // An event that we will trigger when the task completes.
    MapperEvent event;
    // The point in time when we scheduled the task.
    std::chrono::high_resolution_clock::time_point schedTime;
  };
  // queue maintains the current tasks executing on each processor.
  std::map<Processor, std::deque<InFlightTask>> queue;
  // maxInFlightTasks controls how many tasks can execute at a time
  // on a single processor.
  size_t maxInFlightTasks = 1;
  // kase=0: lower than expected; should send stealing requests;
  // kase=1: within the range;
  // kase=2: higher than expected; should permit stealing
  int kase = 0;
  int lower_bound = 2;
  int higher_bound = 10;
  // key: processor id; value: number of stolen tasks
  std::map<long long, int> proc_stolen_tasks;
  int scheduled_tasks = 0; // for assertion
};

DLBMapper::DLBMapper(MapperRuntime *rt, Machine machine, Processor local,
                     const char *mapper_name)
  : DefaultMapper(rt, machine, local, mapper_name)
{
}

void DLBMapper::select_task_options(const MapperContext    ctx,
                                    const Task&            task,
                                          TaskOptions&     output)
//--------------------------------------------------------------------------
{
  output.initial_proc = local_cpus[0];
  output.inline_task = false;
  output.stealable = (task.task_id == COMPUTE_TASK_ID ? true : false); // turn on stealing!
  output.map_locally = map_locally;
#ifdef DEBUG_CTRL_REPL
  if (task.get_depth() == 0)
#else
  if ((total_nodes > 1) && (task.get_depth() == 0))
#endif
    output.replicate = replication_enabled;
  else
    output.replicate = false;
}


void DLBMapper::select_tasks_to_map(const MapperContext ctx,
                         const SelectMappingInput& input,
                               SelectMappingOutput& output)
{
  auto schedTime = std::chrono::high_resolution_clock::now();
  // Maintain an event that we can return in case we don't schedule anything.
  // This event will be used by the runtime to determine when it should ask us
  // to schedule more tasks for mapping. We also maintain a timestamp with the
  // return event. This is so that we choose the earliest scheduled task to wait
  // on, so that we can keep the processors busy.
  MapperEvent returnEvent;
  auto returnTime = std::chrono::high_resolution_clock::time_point::max();
  
  int ready_size = (int) input.ready_tasks.size();
  if (ready_size < lower_bound)
  {
    kase = 0;
  }
  else if (ready_size >= lower_bound && ready_size <= higher_bound)
  {
    kase = 1;
  }
  else
  {
    kase = 2;
  }
  // Schedule all of our available tasks, except tasks with TID_WORKER,
  // to which we'll backpressure.
  for (auto task : input.ready_tasks) {
    bool schedule = true;
    if (std::string(task->get_task_name()) == std::string("compute_task"))
    {
      auto inflight = this->queue[task->target_proc];
      if (inflight.size() == this->maxInFlightTasks) {
        // We've hit the cap, so we can't schedule any more tasks.
        schedule = false;
        // As a heuristic, we'll wait on the first mapper event to
        // finish, as it's likely that one will finish first. We'll also
        // try to get a task that will complete before the current best.
        auto front = inflight.front();
        if (front.schedTime < returnTime) {
          returnEvent = front.event;
          returnTime = front.schedTime;
        }
      } else {
        // Otherwise, we can schedule the task. Create a new event
        // and queue it up on the processor.
        this->queue[task->target_proc].push_back({
          .id = task->get_unique_id(),
          .event = this->runtime->create_mapper_event(ctx),
          .schedTime = schedTime,
        });
        this->scheduled_tasks++;
      }
    }
    else if (std::string(task->get_task_name()) == std::string("last_task"))
    {
      assert(this->scheduled_tasks == num_tasks / num_cpus);
    }
    // Schedule tasks that passed the check.
    if (schedule) {
      output.map_tasks.insert(task);
    }
  }
  // If we don't schedule any tasks for mapping, the runtime needs to know
  // when to ask us again to schedule more things. Return the MapperEvent we
  // selected earlier.
  if (output.map_tasks.size() == 0) {
    assert(returnEvent.exists());
    output.deferral_event = returnEvent;
  }
}

void DLBMapper::default_policy_select_target_processors(MapperContext ctx,
                                                        const Task &task,
                                                        std::vector<Processor> &target_procs)
{
  target_procs.push_back(task.target_proc); // local_cpus[0]
}

void DLBMapper::map_task(const MapperContext ctx,
                const Task& task,
                const MapTaskInput& input,
                MapTaskOutput& output)
{
  DefaultMapper::map_task(ctx, task, input, output);
  if (std::string(task.get_task_name()) == std::string("compute_task"))
  {
    output.task_prof_requests.add_measurement<ProfilingMeasurements::OperationStatus>();
  }
}

void DLBMapper::report_profiling(const MapperContext ctx,
                        const Task& task,
                        const TaskProfilingInfo& input)
{
  // We expect all of our tasks to complete successfully
  auto prof = input.profiling_responses.get_measurement<ProfilingMeasurements::OperationStatus>();
  assert(prof->result == Realm::ProfilingMeasurements::OperationStatus::COMPLETED_SUCCESSFULLY);
  // Clean up after ourselves.
  delete prof;
  // Iterate through the queue and find the event for this task instance.
  std::deque<InFlightTask>& inflight = this->queue[task.target_proc];
  MapperEvent event;
  for (auto it = inflight.begin(); it != inflight.end(); it++) {
    if (it->id == task.get_unique_id()) {
      event = it->event;
      inflight.erase(it);
      break;
    }
  }
  assert(event.exists());
  // Trigger the event so that the runtime knows it's time to schedule
  // some more tasks to map.
  this->runtime->trigger_mapper_event(ctx, event);
}

// We use the non-reentrant serialized model as we want to ensure sole access to
// mapper data structures. If we use the reentrant model, we would have to lock
// accesses to the queue, since we use it interchanged with calls into the runtime.
Mapper::MapperSyncModel DLBMapper::get_mapper_sync_model() const
{
  return SERIALIZED_NON_REENTRANT_MAPPER_MODEL;
}

void DLBMapper::select_steal_targets(const MapperContext         ctx,
                          const SelectStealingInput&  input,
                                SelectStealingOutput& output)
{
  if (kase == 0)
  {
    for (auto p : local_cpus) {
      if (local_proc == p)
      {
        continue;
      }
      output.targets.insert(p);
    }
    for (auto p : remote_cpus) {
      output.targets.insert(p);
    }
  }
  else if (kase == 1)
  {
    return; // do nothing
  }
  else
  {
    assert(kase == 2);
    return;
  }
}

void DLBMapper::permit_steal_request(const MapperContext       ctx,
                          const StealRequestInput&  input,
                               StealRequestOutput& output)
{
  output.stolen_tasks.clear();
  if (kase == 2)
  {
    long long thief_id = (long long) input.thief_proc.id;
    if (proc_stolen_tasks.count(thief_id) == 0)
    {
      proc_stolen_tasks[thief_id] = 1;
    }
    else
    {
      proc_stolen_tasks[thief_id]++;
    }
    if (proc_stolen_tasks[thief_id] <= higher_bound)
    {
      output.stolen_tasks.insert(input.stealable_tasks[0]);
    }
  }
}

static void create_mappers(Machine machine, Runtime *runtime, const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    DLBMapper* mapper = new DLBMapper(runtime->get_mapper_runtime(), machine, *it, "circuit_mapper");
    runtime->replace_default_mapper(mapper, *it);
  }
}

void register_mappers()
{
  Runtime::add_registration_callback(create_mappers);
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(COMPUTE_TASK_ID, "compute_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<compute_task>(registrar, "compute_task");
  }

  {
    TaskVariantRegistrar registrar(LAST_TASK_ID, "last_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<last_task>(registrar, "last_task");
  }

  register_mappers();

  return Runtime::start(argc, argv);
}