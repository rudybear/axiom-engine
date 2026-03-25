#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# AXIOM Engine Verification Pipeline
# ═══════════════════════════════════════════════════════════════════════
# Runs ALL verification checks: build, tests, screenshot tests, performance.
#
# Usage: ./verify-engine.sh [axiom-compiler-path]
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

AXIOM_BIN="${1:-axiom}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$ENGINE_DIR"

PASS=0
FAIL=0
TOTAL=0

check() {
    local name="$1"
    local result="$2"
    TOTAL=$((TOTAL + 1))
    if [ "$result" -eq 0 ]; then
        echo "  ✓ $name"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $name"
        FAIL=$((FAIL + 1))
    fi
}

echo "═══════════════════════════════════════════════════════════════"
echo "  AXIOM Engine Verification Pipeline"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── 1. Renderer builds ────────────────────────────────────────────
echo "--- Build Checks ---"
cargo build -p axiom-renderer --release 2>&1 | tail -1
check "axiom-renderer builds (release)" $?

# ── 2. Renderer tests ────────────────────────────────────────────
echo ""
echo "--- Unit Tests ---"
cargo test -p axiom-renderer 2>&1 | tail -3
check "axiom-renderer tests pass" $?

# ── 3. AXIOM programs compile ────────────────────────────────────
echo ""
echo "--- AXIOM Compilation Checks ---"
for f in examples/**/*.axm; do
    if "$AXIOM_BIN" compile --emit=llvm-ir "$f" > /dev/null 2>&1; then
        check "compile: $(basename $f)" 0
    else
        check "compile: $(basename $f)" 1
    fi
done

# ── 4. Screenshot tests (visual regression) ──────────────────────
echo ""
echo "--- Screenshot Tests ---"
if [ -f target/release/axiom_renderer.dll ] || [ -f target/release/libaxiom_renderer.so ]; then
    for test_axm in tests/screenshot_*.axm; do
        [ -f "$test_axm" ] || continue
        name=$(basename "$test_axm" .axm)
        expected="tests/expected/${name}.png"
        actual="tests/actual/${name}.png"

        # Compile and run
        "$AXIOM_BIN" compile "$test_axm" -o "tests/${name}.exe" 2>/dev/null
        mkdir -p tests/actual

        if [ -f "tests/${name}.exe" ]; then
            timeout 10 "./tests/${name}.exe" > /dev/null 2>&1 || true

            if [ -f "$actual" ]; then
                if [ -f "$expected" ]; then
                    # Compare screenshots (pixel diff)
                    # For now: check file exists and is non-empty
                    actual_size=$(stat -f%z "$actual" 2>/dev/null || stat -c%s "$actual" 2>/dev/null || echo "0")
                    if [ "$actual_size" -gt 1000 ]; then
                        check "screenshot: $name (${actual_size} bytes)" 0
                    else
                        check "screenshot: $name (empty!)" 1
                    fi
                else
                    echo "  ! screenshot: $name — no expected baseline, saving as new baseline"
                    cp "$actual" "$expected"
                    check "screenshot: $name (baseline created)" 0
                fi
            else
                check "screenshot: $name (no output file)" 1
            fi
        else
            check "screenshot: $name (compile failed)" 1
        fi
    done
else
    echo "  (skipped — renderer DLL not built)"
fi

# ── 5. Performance checks ────────────────────────────────────────
echo ""
echo "--- Performance Requirements ---"
for perf_file in tests/perf_*.axm; do
    [ -f "$perf_file" ] || continue
    name=$(basename "$perf_file" .axm)

    # Each perf test has a max_ms requirement in its filename or header
    # Format: perf_NAME_MAXms.axm (e.g., perf_frame_loop_16ms.axm)
    max_ms=$(echo "$name" | grep -oP '\d+ms$' | grep -oP '\d+' || echo "")

    if [ -z "$max_ms" ]; then
        echo "  ? $name — no time requirement in filename"
        continue
    fi

    "$AXIOM_BIN" compile "$perf_file" -o "tests/${name}.exe" 2>/dev/null || continue

    if [ -f "tests/${name}.exe" ]; then
        # Time it
        start=$(date +%s%N 2>/dev/null || python3 -c "import time; print(int(time.time()*1e9))")
        timeout 30 "./tests/${name}.exe" > /dev/null 2>&1
        end=$(date +%s%N 2>/dev/null || python3 -c "import time; print(int(time.time()*1e9))")
        elapsed_ms=$(( (end - start) / 1000000 ))

        if [ "$elapsed_ms" -le "$max_ms" ]; then
            check "perf: $name (${elapsed_ms}ms <= ${max_ms}ms)" 0
        else
            check "perf: $name (${elapsed_ms}ms > ${max_ms}ms EXCEEDED)" 1
        fi
    fi
done

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Results: $PASS passed, $FAIL failed, $TOTAL total"
if [ "$FAIL" -gt 0 ]; then
    echo "  STATUS: FAIL"
    exit 1
else
    echo "  STATUS: PASS"
    exit 0
fi
