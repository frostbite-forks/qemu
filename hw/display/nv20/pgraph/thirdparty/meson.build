nv20_vsh_cpu_opts = cmake.subproject_options()
nv20_vsh_cpu_opts.add_cmake_defines({
  'CMAKE_OSX_ARCHITECTURES': cmake_macos_arch,
  'nv20_vsh_cpu_UNIT_TEST': 'OFF',
  })
nv20_vsh_cpu_subproj = cmake.subproject('nv20_vsh_cpu', options: nv20_vsh_cpu_opts)
nv20_vsh_cpu = declare_dependency(include_directories: nv20_vsh_cpu_subproj.include_directories('nv20_vsh_emulator'),
                                  link_with: [
                                    nv20_vsh_cpu_subproj.target('nv20_vsh_emulator'),
                                    nv20_vsh_cpu_subproj.target('nv20_vsh_cpu'),
                                    nv20_vsh_cpu_subproj.target('nv20_vsh_disassembler'),
                                    ])

libgloffscreen = static_library('libgloffscreen',
                          sources: files('gloffscreen/common.c', 'gloffscreen/sdl.c'),
                          dependencies: sdl)
gloffscreen = declare_dependency(link_with: libgloffscreen,
                           include_directories: ['gloffscreen'])
