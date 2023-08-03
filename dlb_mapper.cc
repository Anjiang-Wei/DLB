/* Copyright 2022 Stanford University
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

#include "dlb_mapper.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <map>
#include <vector>

#include "mappers/default_mapper.h"
#include "mappers/logging_wrapper.h"
#include <chrono>
uint64_t timeSinceEpochMillisec() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

using namespace Legion;
using namespace Legion::Mapping;

///
/// Mapper
///

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
  // int num_ready_for_steal = 4;
  // bool ready_for_steal = false;
  int kase = 0;// 0: lower than expected; should send stealing requests;
  // 1: within the range;
  // 2: higher than expected; should permit stealing
  int lower_bound = 2;
  int higher_bound = 10;
  std::map<int, int> proc_stolen_tasks;
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
  // log_mapper.spew("Default select_task_options in %s", get_mapper_name());
  // printf("initial_proc = %d\n", local_cpus[0].id);
  output.initial_proc = local_cpus[0];// default_policy_select_initial_processor(ctx, task);
  output.inline_task = false;
  // printf("stealable turned on in select_task_options %s\n", task.get_task_name());
  output.stealable = true; // turn on stealing!
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
  // std::cout << "timestamp for select_tasks_to_map: " << timeSinceEpochMillisec() << std::endl;
  // Record when we are scheduling tasks.
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

    printf("select_tasks_to_map for f, input.ready_tasks.size() = %d, kase = %d, in proc_id = %d\n",
        (int) input.ready_tasks.size(), kase, local_proc.id);

  // Schedule all of our available tasks, except tasks with TID_WORKER,
  // to which we'll backpressure.
  for (auto task : input.ready_tasks) {
    bool schedule = true;
    if (std::string(task->get_task_name()) == std::string("f")) {
    // if (task->task_id == TID_WORKER && this->enableBackPressure) {
      // See how many tasks we have in flight.
      // if (input.ready_tasks.size() >= num_ready_for_steal)
      // {
      //   ready_for_steal = true;
      // }
      auto inflight = this->queue[task->target_proc];
      if (inflight.size() == this->maxInFlightTasks) {
        // printf("enable backpressure!\n");
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
      }
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
  target_procs.push_back(task.target_proc);
  // for (auto p : local_cpus) {
  //   target_procs.push_back(p);
  // }
}

void DLBMapper::map_task(const MapperContext ctx,
                const Task& task,
                const MapTaskInput& input,
                MapTaskOutput& output)
{
  DefaultMapper::map_task(ctx, task, input, output);
  // We need to know when the TID_WORKER tasks complete, so we'll ask the runtime
  // to give us profiling information when they complete.
  if (std::string(task.get_task_name()) == std::string("f")) {
    std::deque<InFlightTask>& inflight = this->queue[task.target_proc];
    // printf("map_task for task %s %d, inflight.size() = %d\n",
    //         task.get_task_name(), (int) task.get_unique_id(), (int) inflight.size());
    output.task_prof_requests.add_measurement<ProfilingMeasurements::OperationStatus>();
  }
}

void DLBMapper::report_profiling(const MapperContext ctx,
                        const Task& task,
                        const TaskProfilingInfo& input)
{
  // Only TID_WORKER tasks should have profiling information.
  assert(std::string(task.get_task_name()) == std::string("f"));
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
  // int kase = 0;// 0: lower than expected; should send stealing requests;
  // 1: within the range;
  // 2: higher than expected; should permit stealing
  printf("select_steal_targets from cpu %d\n", local_proc.id);
  if (kase == 0)
  {
    for (auto p : local_cpus) {
      if (local_proc == p)
      {
        continue;
      }
      output.targets.insert(p);
    }
    printf("select_steal_targets is actually sending requests from cpu %d\n", local_proc.id);
    // kase = 1;
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
  // std::cout << "timestamp for permit_steal_request: " << timeSinceEpochMillisec() << std::endl;
  output.stolen_tasks.clear();
  if (kase == 2)
  {
    int thief_id = (int) input.thief_proc.id;
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
      printf("permit_steal_request works %d out of %d stealable tasks, from cpu %d sending to cpu %d\n",
         proc_stolen_tasks[thief_id], (int) input.stealable_tasks.size(),
         (int) local_proc.id, (int) input.thief_proc.id);
    }
  }
}

static void create_mappers(Machine machine, Runtime *runtime, const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    DLBMapper* mapper = new DLBMapper(runtime->get_mapper_runtime(), machine, *it, "circuit_mapper");
    runtime->replace_default_mapper(new LoggingWrapper(mapper), *it);
  }
  printf("create mappers have replaced %d mappers\n", (int) local_procs.size());
}

void register_mappers()
{
  Runtime::add_registration_callback(create_mappers);
}