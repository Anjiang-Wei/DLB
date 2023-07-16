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
  template<int DIM>
  void default_decompose_points(const DomainT<DIM,coord_t> &point_space,
                                const std::vector<Processor> &targets,
                                const Point<DIM,coord_t> &num_blocks,
                                bool recurse, bool stealable,
                                std::vector<TaskSlice> &slices);
  void default_slice_task(const Task &task,
                          const std::vector<Processor> &local,
                          const std::vector<Processor> &remote,
                          const SliceTaskInput& input,
                                SliceTaskOutput &output,
      std::map<Domain,std::vector<TaskSlice> > &cached_slices);
  void slice_task(const MapperContext      ctx,
                  const Task&              task,
                  const SliceTaskInput&    input,
                        SliceTaskOutput&   output) override;
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
  output.initial_proc = default_policy_select_initial_processor(ctx, task);
  output.inline_task = false;
  printf("stealable turned on in select_task_options %s\n", task.get_task_name());
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
  // Record when we are scheduling tasks.
  auto schedTime = std::chrono::high_resolution_clock::now();

  // Maintain an event that we can return in case we don't schedule anything.
  // This event will be used by the runtime to determine when it should ask us
  // to schedule more tasks for mapping. We also maintain a timestamp with the
  // return event. This is so that we choose the earliest scheduled task to wait
  // on, so that we can keep the processors busy.
  MapperEvent returnEvent;
  auto returnTime = std::chrono::high_resolution_clock::time_point::max();

  // Schedule all of our available tasks, except tasks with TID_WORKER,
  // to which we'll backpressure.
  for (auto task : input.ready_tasks) {
    bool schedule = true;
    if (std::string(task->get_task_name()) == std::string("f")) {
      printf("select_tasks_to_map for f %d %s, input.ready_tasks.size() = %d\n", (int) task->get_unique_id(),
            task->stealable ? "stealable" : "non-stealable", (int) input.ready_tasks.size());
    // if (task->task_id == TID_WORKER && this->enableBackPressure) {
      // See how many tasks we have in flight.
      auto inflight = this->queue[task->target_proc];
      if (inflight.size() == this->maxInFlightTasks) {
        printf("enable backpressure!\n");
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
    printf("map_task for task %s %d, inflight.size() = %d\n",
            task.get_task_name(), (int) task.get_unique_id(), (int) inflight.size());
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

void DLBMapper::default_policy_select_target_processors(MapperContext ctx,
                                                        const Task &task,
                                                        std::vector<Processor> &target_procs)
{
  target_procs.push_back(task.target_proc);
}

void DLBMapper::slice_task(const MapperContext      ctx,
                          const Task&              task,
                          const SliceTaskInput&    input,
                                SliceTaskOutput&   output)
//--------------------------------------------------------------------------
{
  // log_mapper.spew("Default slice_task in %s", get_mapper_name());

  std::vector<VariantID> variants;
  runtime->find_valid_variants(ctx, task.get_unique_id(), variants);
  /* find if we have a procset variant for task */
  for(unsigned i = 0; i < variants.size(); i++)
  {
    const ExecutionConstraintSet exset =
        runtime->find_execution_constraints(ctx, task.get_unique_id(), variants[i]);
    if(exset.processor_constraint.can_use(Processor::PROC_SET)) {

        // Before we do anything else, see if it is in the cache
        std::map<Domain,std::vector<TaskSlice> >::const_iterator finder =
          procset_slices_cache.find(input.domain);
        if (finder != procset_slices_cache.end()) {
                output.slices = finder->second;
                return;
        }

      output.slices.resize(input.domain.get_volume());
      unsigned idx = 0;
      Rect<1> rect = input.domain;
      for (PointInRectIterator<1> pir(rect); pir(); pir++, idx++)
      {
        Rect<1> slice(*pir, *pir);
        output.slices[idx] = TaskSlice(slice,
          remote_procsets[idx % remote_cpus.size()],
          false/*recurse*/, false/*stealable*/);
      }

      // Save the result in the cache
      procset_slices_cache[input.domain] = output.slices;
      return;
    }
  }


  // Whatever kind of processor we are is the one this task should
  // be scheduled on as determined by select initial task
  Processor::Kind target_kind =
    task.must_epoch_task ? local_proc.kind() : task.target_proc.kind();
  switch (target_kind)
  {
    case Processor::LOC_PROC:
      {
        default_slice_task(task, local_cpus, remote_cpus,
                            input, output, cpu_slices_cache);
        break;
      }
    case Processor::TOC_PROC:
      {
        default_slice_task(task, local_gpus, remote_gpus,
                            input, output, gpu_slices_cache);
        break;
      }
    case Processor::IO_PROC:
      {
        default_slice_task(task, local_ios, remote_ios,
                            input, output, io_slices_cache);
        break;
      }
    case Processor::PY_PROC:
      {
        default_slice_task(task, local_pys, remote_pys,
                            input, output, py_slices_cache);
        break;
      }
    case Processor::PROC_SET:
      {
        default_slice_task(task, local_procsets, remote_procsets,
                            input, output, procset_slices_cache);
        break;
      }
    case Processor::OMP_PROC:
      {
        default_slice_task(task, local_omps, remote_omps,
                            input, output, omp_slices_cache);
        break;
      }
    default:
      assert(false); // unimplemented processor kind
  }
}

void DLBMapper::default_slice_task(const Task &task,
                                      const std::vector<Processor> &local,
                                      const std::vector<Processor> &remote,
                                      const SliceTaskInput& input,
                                            SliceTaskOutput &output,
                  std::map<Domain,std::vector<TaskSlice> > &cached_slices)
//--------------------------------------------------------------------------
{
  // Before we do anything else, see if it is in the cache
  std::map<Domain,std::vector<TaskSlice> >::const_iterator finder =
    cached_slices.find(input.domain);
  if (finder != cached_slices.end()) {
    output.slices = finder->second;
    return;
  }

  // The two-level decomposition doesn't work so for now do a
  // simple one-level decomposition across all the processors.
  // Unless we're doing same address space mapping or this is
  // a task in a control-replicated root task
  Machine::ProcessorQuery all_procs(machine);
  all_procs.only_kind(local[0].kind());
  if (((task.tag & SAME_ADDRESS_SPACE) != 0) || same_address_space)
all_procs.local_address_space();
  else if (replication_enabled && (task.get_depth() == 1))
  {
    // Check to see if the parent task is control replicated
    const Task *parent = task.get_parent_task();
    if ((parent != NULL) && (parent->get_total_shards() > 1))
      all_procs.local_address_space();
  }
  std::vector<Processor> procs(all_procs.begin(), all_procs.end());

  switch (input.domain.get_dim())
  {
#define BLOCK(DIM) \
    case DIM: \
      { \
        DomainT<DIM,coord_t> point_space = input.domain; \
        Point<DIM,coord_t> num_blocks = \
          default_select_num_blocks<DIM>(procs.size(), point_space.bounds); \
        DLBMapper::default_decompose_points<DIM>(point_space, procs, \
              num_blocks, false/*recurse*/, \
              stealing_enabled, output.slices); \
        break; \
      }
    LEGION_FOREACH_N(BLOCK)
#undef BLOCK
    default: // don't support other dimensions right now
      assert(false);
  }
  // Save the result in the cache
  cached_slices[input.domain] = output.slices;
}

template<int DIM>
/*static*/ void DLBMapper::default_decompose_points(
                        const DomainT<DIM,coord_t> &point_space,
                        const std::vector<Processor> &targets,
                        const Point<DIM,coord_t> &num_blocks,
                        bool recurse, bool stealable,
                        std::vector<TaskSlice> &slices)
//--------------------------------------------------------------------------
{
  stealable = true; // always stealable
  Point<DIM,coord_t> zeroes;
  for (int i = 0; i < DIM; i++)
    zeroes[i] = 0;
  Point<DIM,coord_t> ones;
  for (int i = 0; i < DIM; i++)
    ones[i] = 1;
  Point<DIM,coord_t> num_points = 
    point_space.bounds.hi - point_space.bounds.lo + ones;
  Rect<DIM,coord_t> blocks(zeroes, num_blocks - ones);
  size_t next_index = 0;
  slices.reserve(blocks.volume());
  for (PointInRectIterator<DIM> pir(blocks); pir(); pir++) {
    Point<DIM,coord_t> block_lo = *pir;
    Point<DIM,coord_t> block_hi = *pir + ones;
    Point<DIM,coord_t> slice_lo =
      num_points * block_lo / num_blocks + point_space.bounds.lo;
    Point<DIM,coord_t> slice_hi = 
      num_points * block_hi / num_blocks + point_space.bounds.lo - ones;
    // Construct a new slice space based on the new bounds 
    // and any existing sparsity map, tighten if necessary
    DomainT<DIM,coord_t> slice_space;
    slice_space.bounds.lo = slice_lo;
    slice_space.bounds.hi = slice_hi;
    slice_space.sparsity = point_space.sparsity;
    if (!slice_space.dense())
      slice_space = slice_space.tighten();
    if (slice_space.volume() > 0) {
      TaskSlice slice;
      slice.domain = slice_space;
      slice.proc = targets[next_index++ % targets.size()];
      slice.recurse = recurse;
      printf("stealable turned on in slice_task\n");
      slice.stealable = stealable;
      slices.push_back(slice);
    }
  }
}

void DLBMapper::select_steal_targets(const MapperContext         ctx,
                          const SelectStealingInput&  input,
                                SelectStealingOutput& output)
{
  printf("select_steal_targets invoked\n");
  output.targets.clear();
  for (auto p : local_cpus) {
    if (local_proc == p) continue;
    output.targets.insert(p);
    printf("select_steal_targets insert one local cpu processor\n");
  }

}

void DLBMapper::permit_steal_request(const MapperContext       ctx,
                          const StealRequestInput&  input,
                               StealRequestOutput& output)
{
  printf("permit_steal_request invoked\n");
  output.stolen_tasks.clear();
  // Iterate over stealable tasks
  for (auto task : input.stealable_tasks) {
    output.stolen_tasks.insert(task);
    printf("permit_steal_request insert one stealable task\n");
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
}

void register_mappers()
{
  Runtime::add_registration_callback(create_mappers);
}