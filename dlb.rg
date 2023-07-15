import "regent"

local format = require("std/format")

-- Compile and link dlb_mapper.cc
local cmapper
do
  local root_dir = arg[0]:match(".*/") or "./"

  local include_path = ""
  local include_dirs = terralib.newlist()
  include_dirs:insert("-I")
  include_dirs:insert(root_dir)
  for path in string.gmatch(os.getenv("INCLUDE_PATH"), "[^;]+") do
    include_path = include_path .. " -I " .. path
    include_dirs:insert("-I")
    include_dirs:insert(path)
  end

  local mapper_cc = root_dir .. "dlb_mapper.cc"
  local mapper_so
  if os.getenv('OBJNAME') then
    local out_dir = os.getenv('OBJNAME'):match('.*/') or './'
    mapper_so = out_dir .. "libdlb_mapper.so"
  elseif os.getenv('SAVEOBJ') == '1' then
    mapper_so = root_dir .. "libdlb_mapper.so"
  else
    mapper_so = os.tmpname() .. ".so" -- root_dir .. "dlb_mapper.so"
  end
  local cxx = os.getenv('CXX') or 'c++'

  local cxx_flags = os.getenv('CXXFLAGS') or ''
  -- cxx_flags = cxx_flags .. " -O2 -Wall -Werror"
  cxx_flags = cxx_flags .. " -O2 -Wall"
  if os.execute('test "$(uname)" = Darwin') == 0 then
    cxx_flags =
      (cxx_flags ..
         " -dynamiclib -single_module -undefined dynamic_lookup -fPIC")
  else
    cxx_flags = cxx_flags .. " -shared -fPIC"
  end

  local cmd = (cxx .. " " .. cxx_flags .. " " .. include_path .. " " ..
                 mapper_cc .. " -o " .. mapper_so)
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile " .. mapper_cc)
    assert(false)
  end
  regentlib.linklibrary(mapper_so)
  cmapper = terralib.includec("dlb_mapper.h", include_dirs)
end

local c = regentlib.c
local std = terralib.includec("stdlib.h")
local cmath = terralib.includec("math.h")
local cstring = terralib.includec("string.h")
local unistd = terralib.includec("unistd.h")
rawset(_G, "drand48", std.drand48)
rawset(_G, "srand48", std.srand48)
rawset(_G, "ceil", cmath.ceil)

---  DBL Start

task f(i: int)
-- just wait for a random number of seconds
  unistd.sleep(i % 4)
  format.println("DLB {} wait for {} seconds", i, i % 4)
end

task toplevel()
  var num_points = 40
  -- __demand(__index_launch)
  __forbid(__index_launch)
  for i = 0, num_points do
    f(i)
  end
  format.println("done")
end

---- Compile

if os.getenv('SAVEOBJ') == '1' then
  local root_dir = arg[0]:match(".*/") or "./"
  local out_dir = (os.getenv('OBJNAME') and os.getenv('OBJNAME'):match('.*/')) or root_dir
  local link_flags = terralib.newlist({"-L" .. out_dir, "-ldlb_mapper", "-lm"})

  if os.getenv('STANDALONE') == '1' then
    os.execute('cp ' .. os.getenv('LG_RT_DIR') .. '/../bindings/regent/' ..
        regentlib.binding_library .. ' ' .. out_dir)
  end

  local exe = os.getenv('OBJNAME') or "dlb"
  regentlib.saveobj(toplevel, exe, "executable", cmapper.register_mappers, link_flags)
else
  regentlib.start(toplevel, cmapper.register_mappers)
end