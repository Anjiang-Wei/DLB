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
private:
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
  runtime->find_valid_variants(ctx, task.task_id, variants);
  /* find if we have a procset variant for task */
  for(unsigned i = 0; i < variants.size(); i++)
  {
    const ExecutionConstraintSet exset =
        runtime->find_execution_constraints(ctx, task.task_id, variants[i]);
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