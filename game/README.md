# Asteroid Field

A 2D top-down space survival game built on the AXIOM engine's software renderer.

## How it works

- **Ship** is rendered as a colored triangle (cyan tip, blue wings) via `renderer_draw_triangles`
- **Asteroids** are rendered as colored 2x2 pixel points via `renderer_draw_points`
- **Collision** uses circle-circle distance testing
- **No GPU DLL needed** -- uses the Win32 software rasterizer built into `axiom_rt.c`

## Controls

| Key   | Action     |
|-------|------------|
| W     | Move up    |
| A     | Move left  |
| S     | Move down  |
| D     | Move right |
| Escape| Quit       |

## Build and Run

```bash
# From the axiom-engine root directory:
axiom compile game/asteroid_field.axm -o asteroid.exe

# Run the game:
./asteroid.exe
```

No external DLLs are required. The software renderer, input handling, and audio
are all provided by the C runtime (`axiom_rt.c`) which is statically linked.

## Gameplay

- Dodge asteroids that spawn from the screen edges and drift inward
- Difficulty increases over time: asteroids spawn faster and move quicker
- Your score is the number of frames survived
- On collision the screen turns dark red and the final score is printed

## Architecture

The game uses SOA (Struct of Arrays) layout for asteroid data, matching the
AXIOM engine's cache-friendly data patterns:

```
ast_x[N], ast_y[N]       -- positions (f64)
ast_vx[N], ast_vy[N]     -- velocities (f64)
ast_radius[N]             -- collision radii (f64)
ast_alive[N]              -- active flags (i32)
ast_colors[N]             -- packed 0xRRGGBB colors (i32)
```

All asteroid data is heap-allocated once at startup. Zero per-frame allocations.

## Performance test

A headless performance test is available at `tests/perf_game_logic_100ms.axm`.
It runs 1000 frames of pure game logic (no rendering) and must complete in
under 100ms.
