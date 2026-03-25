# AXIOM Engine Test Suite

## Test Types

### 1. Unit Tests (Rust)
```bash
cargo test -p axiom-renderer
```
Tests the renderer internals: camera math, vertex generation, pipeline creation.

### 2. Screenshot Tests (`tests/screenshot_*.axm`)
Visual regression tests. Each compiles an AXIOM program, renders a scene, captures a screenshot, and compares against a baseline.

| Test | What it verifies | Baseline |
|------|-----------------|----------|
| `screenshot_pbr_helmet` | PBR DamagedHelmet renders with correct materials | `tests/expected/screenshot_pbr_helmet.png` |

**Running:** `bash .pipeline/scripts/verify-engine.sh`

First run creates the baseline. Subsequent runs compare against it.

### 3. Performance Tests (`tests/perf_*_NNNms.axm`)
Each test must complete within the time budget in its filename.

| Test | Budget | What it measures |
|------|--------|-----------------|
| `perf_frame_loop_1000ms` | 1000ms | 1000 empty frames (GPU overhead) |
| `perf_gltf_load_5000ms` | 5000ms | glTF load + 10 PBR render frames |
| `perf_collision_100ms` | 100ms | 1M sphere-sphere collision checks |

**Convention:** `perf_NAME_NNNms.axm` — the `NNN` is the max allowed milliseconds.

### 4. Integration Tests
The game itself (`game/asteroid_field.axm`) serves as an integration test. If it compiles and runs for 100 frames without crashing, the engine works.

## Running All Tests

```bash
# Full verification pipeline
bash .pipeline/scripts/verify-engine.sh

# Just Rust tests
cargo test -p axiom-renderer

# Just screenshot tests (requires renderer DLL + AXIOM compiler)
# Screenshot tests need a GPU — skip on headless CI
```

## Adding New Tests

### Screenshot test:
1. Create `tests/screenshot_THING.axm` — render scene, call `gpu_screenshot`
2. Run once to create baseline: `tests/expected/screenshot_THING.png`
3. Commit the baseline
4. CI compares future renders against the baseline

### Performance test:
1. Create `tests/perf_THING_NNNms.axm` — NNN is the time budget
2. The verify script times execution and fails if budget exceeded
3. Budget should be 2-3x the expected time (for CI variance)
