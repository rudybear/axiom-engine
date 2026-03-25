# AXIOM Engine

Game engine and Vulkan renderer built on the [AXIOM compiler](https://github.com/rudybear/axiom).

**AXIOM** handles all CPU logic (game loop, physics, ECS, AI). **Lux** handles GPU shaders (PBR, particles, post-processing). **wgpu** provides the Vulkan/DX12/Metal rendering backend.

## Features

- **PBR Renderer** — Cook-Torrance BRDF, normal mapping, metallic-roughness workflow
- **glTF Loading** — meshes, materials, textures from standard glTF/GLB files
- **Procedural Meshes** — cube, sphere generation (no external models needed)
- **Multi-Light** — up to 8 point lights with inverse-square attenuation
- **Instanced Rendering** — draw thousands of objects efficiently
- **Input System** — keyboard + mouse via winit
- **ECS Library** — Entity-Component-System written in pure AXIOM
- **Lux Integration** — load SPIR-V shaders compiled by the Lux shader language
- **Screenshot Capture** — `gpu_screenshot` for automated testing

## Quick Start

```bash
# Prerequisites: Rust, clang, AXIOM compiler (from github.com/rudybear/axiom)

# Build the renderer DLL
cargo build -p axiom-renderer --release

# Compile an AXIOM game program
axiom compile examples/vulkan/pbr_scene.axm -o pbr_scene.exe

# Copy DLL next to exe
cp target/release/axiom_renderer.dll .

# Run
./pbr_scene.exe
```

## Structure

```
axiom-engine/
├── axiom-renderer/          # Rust wgpu renderer (cdylib)
│   ├── src/
│   │   ├── lib.rs           # C ABI exports (30+ functions)
│   │   ├── renderer.rs      # wgpu setup, frame loop, draw commands
│   │   ├── pbr.rs           # PBR pipeline + Cook-Torrance shader
│   │   ├── gltf_load.rs     # glTF/GLB mesh + texture loading
│   │   ├── camera.rs        # View/projection matrices
│   │   └── lux_shaders.rs   # SPIR-V shader loading for Lux
│   └── Cargo.toml
├── lib/
│   └── ecs.axm              # ECS library (pure AXIOM)
├── examples/
│   ├── vulkan/              # PBR scene, triangle, rotating model
│   ├── particle_galaxy/     # 10K particle simulation
│   └── killer_demo/         # GPU + input + ECS + audio demo
├── game/                    # Asteroid Field game (WIP)
├── data/                    # glTF models
├── lux/                     # Lux shader language (submodule)
└── docs/                    # Engine documentation
```

## GPU API (AXIOM builtins)

```axiom
// Lifecycle
let gpu: ptr[i32] = gpu_init(1280, 720, "My Game");
gpu_shutdown(gpu);

// Frame loop
while gpu_should_close(gpu) == 0 {
    gpu_begin_frame(gpu);
    gpu_render(gpu);
    gpu_end_frame(gpu);
}

// Scene
let scene: i32 = gpu_load_gltf(gpu, "model.glb");
gpu_set_camera(gpu, eye_x, eye_y, eye_z, target_x, target_y, target_z, fov);

// Lighting
gpu_add_light(gpu, x, y, z, r, g, b, intensity);

// Input
let w: i32 = is_key_down(87);  // W key
let mx: i32 = get_mouse_x();

// Screenshot
gpu_screenshot(gpu, "frame.png");
```

## Depends On

- [AXIOM Compiler](https://github.com/rudybear/axiom) — compiles .axm to native binaries
- [Lux Shader Language](https://github.com/rudybear/lux) — compiles .lux to SPIR-V

## License

MIT
