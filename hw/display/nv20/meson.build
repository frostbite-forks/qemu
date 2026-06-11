nv20_ss = ss.source_set()

nv20_core = files(
  'nv20.c',
  'nv20_pgraph_settings.c',
  'nv20_xemu_compat.c',
  'pfifo.c',
  'pmc.c',
  'pbus.c',
  'pvideo.c',
  'ptimer.c',
  'pcrtc.c',
  'pramdac.c',
  'prmcio.c',
  'prmdio.c',
  'prmvio.c',
  'pfb.c',
  'user.c',
  'stubs.c',
)

nv20_pgraph_core = files(
  'pgraph/pgraph.c',
  'pgraph/profile.c',
  'pgraph/rdi.c',
  'pgraph/s3tc.c',
  'pgraph/swizzle.c',
  'pgraph/texture.c',
  'pgraph/vertex.c',
)

nv20_pgraph_glsl = files(
  'pgraph/glsl/common.c',
  'pgraph/glsl/geom.c',
  'pgraph/glsl/psh.c',
  'pgraph/glsl/shaders.c',
  'pgraph/glsl/vsh.c',
  'pgraph/glsl/vsh-ff.c',
  'pgraph/glsl/vsh-prog.c',
)

nv20_pgraph_null = files('pgraph/null/renderer.c')

nv20_gloffscreen = files(
  'pgraph/thirdparty/gloffscreen/common.c',
  'pgraph/thirdparty/gloffscreen/sdl.c',
)

# nv2a_vsh_cpu — vertex shader CPU emulation (Unlicense)
nv20_vsh_cpu = static_library('nv2a_vsh_cpu',
  sources: files(
    'thirdparty/nv2a_vsh_cpu/src/nv2a_vsh_cpu.c',
    'thirdparty/nv2a_vsh_cpu/src/nv2a_vsh_emulator.c',
    'thirdparty/nv2a_vsh_cpu/src/nv2a_vsh_emulator_execution_state.c',
    'thirdparty/nv2a_vsh_cpu/src/nv2a_vsh_disassembler.c',
  ),
  include_directories: include_directories('thirdparty/nv2a_vsh_cpu/src'),
  c_args: ['-Wno-unused-function'],
)

nv20_vsh_dep = declare_dependency(
  link_with: nv20_vsh_cpu,
  include_directories: include_directories('thirdparty/nv2a_vsh_cpu/src'),
)

nv20_gloffscreen_dep = declare_dependency(
  include_directories: include_directories('pgraph/thirdparty/gloffscreen'),
)

nv20_ss.add(nv20_core)
nv20_ss.add(nv20_pgraph_core)
nv20_ss.add(nv20_pgraph_glsl)
nv20_ss.add(nv20_pgraph_null)
nv20_ss.add(nv20_vsh_dep)
nv20_ss.add(nv20_gloffscreen_dep)

nv20_metal_native = host_machine.system() == 'darwin'

sdl2 = dependency('sdl2', required: false)
if sdl2.found() and not nv20_metal_native
  nv20_ss.add(nv20_gloffscreen)
  nv20_ss.add(sdl2)
endif

# OpenGL offscreen renderer (non-Darwin fallback)
if host_machine.system() != 'darwin' and opengl.found() and sdl2.found()
  nv20_pgraph_gl = files(
    'pgraph/gl/blit.c',
    'pgraph/gl/debug.c',
    'pgraph/gl/display.c',
    'pgraph/gl/draw.c',
    'pgraph/gl/gpuprops.c',
    'pgraph/gl/renderer.c',
    'pgraph/gl/reports.c',
    'pgraph/gl/shaders.c',
    'pgraph/gl/surface.c',
    'pgraph/gl/texture.c',
    'pgraph/gl/vertex.c',
  )
  nv20_ss.add(nv20_pgraph_gl)
  nv20_ss.add(opengl)
endif

# Metal renderer (Darwin) — native Metal backend, no OpenGL/epoxy
if nv20_metal_native
  nv20_metal_fw = dependency('appleframeworks',
                             modules: ['Metal', 'QuartzCore'],
                             required: false)
  if nv20_metal_fw.found()
    nv20_ss.add(files(
      'pgraph/metal/renderer.m',
      'pgraph/metal/display.m',
      'pgraph/metal/draw.m',
      'pgraph/metal/surface.m',
      'pgraph/metal/texture.m',
      'pgraph/metal/shaders.m',
      'pgraph/metal/vertex.m',
      'pgraph/metal/blit.m',
      'pgraph/metal/reports.m',
      'pgraph/metal/gpuprops.m',
      'pgraph/metal/metal_backend.m',
      'pgraph/metal/metal_glsl.m',
      'pgraph/metal/metal_formats.c',
      'pgraph/metal/metal_display_native.m',
    ))
    nv20_ss.add(nv20_metal_fw)
  endif
endif
