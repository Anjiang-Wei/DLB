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
private:
};

DLBMapper::DLBMapper(MapperRuntime *rt, Machine machine, Processor local,
                     const char *mapper_name)
  : DefaultMapper(rt, machine, local, mapper_name)
{
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