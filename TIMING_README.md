# Timing Instrumentation (trace branch)

This branch implements **Approach 1** (Manual Instrumentation) from `PROFILING_PLAN.md` to measure where computation time is spent in fkpt.

## Quick Start

### Basic Timing (recommended)

Run with `chatty=1` to see timing breakdown:

```bash
./fkpt chatty=1 Om=0.3 h=0.7 model=HS fR0=1.0e-6 suffix=_test zout=0.5 fnamePS=pkl_z05.dat
```

### Detailed Timing

Run with `chatty=2` for per-ki Q-loop and R-loop timing (verbose):

```bash
./fkpt chatty=2 Om=0.3 h=0.7 model=HS fR0=1.0e-6 suffix=_test zout=0.5 fnamePS=pkl_z05.dat
```

## Output Explanation

### Level 1: MainLoop Breakdown (chatty >= 1)

At the end of the run, you'll see:

```
======================== TIMING BREAKDOWN ========================
global_variables:           0.001 s (  0.1%)
compute_kfunctions:        45.234 s ( 96.8%)
compute_rsdmultipoles:      1.234 s (  2.6%)
write:                      0.012 s (  0.0%)
free_variables:             0.001 s (  0.0%)
==================================================================
TOTAL MainLoop time:        46.482 s
==================================================================
```

**What this tells you:**
- Which of the 5 main stages dominates runtime
- Expected: `compute_kfunctions` should be >90%

### Level 2: compute_kfunctions Breakdown (chatty >= 1)

During the run, after k-functions complete:

```
--- compute_kfunctions breakdown ---
  Initialization:        0.150 s (  0.3%)
  sigma8 calculation:    0.080 s (  0.2%)
  sigma constants:       0.120 s (  0.3%)
  k-functions loop:     44.884 s ( 99.2%)
  TOTAL:                45.234 s
------------------------------------
```

**What this tells you:**
- Confirms k-functions loop dominates
- sigma8/sigma constants are one-time costs
- Expected: k-functions loop >95% of compute_kfunctions time

### Level 3: Per-k Progress (chatty >= 1)

During k-functions loop, progress reports every 20 iterations:

```
  k[  1/120]: k= 0.001000, time=  0.375 s, avg=  0.375 s/k
  k[ 20/120]: k= 0.002512, time=  0.382 s, avg=  0.378 s/k
  k[ 40/120]: k= 0.006310, time=  0.380 s, avg=  0.379 s/k
  k[ 60/120]: k= 0.015849, time=  0.381 s, avg=  0.379 s/k
  k[ 80/120]: k= 0.039811, time=  0.377 s, avg=  0.379 s/k
  k[100/120]: k= 0.100000, time=  0.378 s, avg=  0.379 s/k
  k[120/120]: k= 0.500000, time=  0.376 s, avg=  0.378 s/k
  Total k-loop time:   45.384 s, avg per k:   0.378 s
```

**What this tells you:**
- Time per k-value (should be roughly constant)
- Running average time per k
- Total time for all k-values
- Any outliers (if some k-values take much longer)

### Level 4: Q-loop and R-loop Breakdown (chatty >= 2)

**Warning: Very verbose!** Prints for each of 120 k-values:

```
    ki= 0.001000: Q-loop= 0.19s, R-loop= 0.18s
    ki= 0.001259: Q-loop= 0.19s, R-loop= 0.18s
    ...
```

**What this tells you:**
- Whether Q-loop (P22 terms) or R-loop (P13 terms) dominates
- Expected: Both should be roughly equal (~50% each)

## What to Look For

### Expected Results (with default parameters):

1. **Total runtime**: ~30-60 seconds on modern CPU
2. **compute_kfunctions**: >90% of total time
3. **k-functions loop**: >95% of compute_kfunctions time
4. **Time per k-value**: ~0.25-0.5 seconds (depends on CPU)
5. **Consistency**: Time per k should be roughly constant

### Red Flags:

- **Highly variable per-k times**: May indicate cache issues or system interference
- **One stage unexpectedly slow**: Could indicate inefficient algorithm or I/O bottleneck
- **Much slower than expected**: Check system load, CPU frequency scaling, or thermal throttling

## Profiling Different Configurations

Test how runtime scales with parameters:

### Vary k-grid resolution (should scale linearly):

```bash
# Half resolution (should be ~2× faster)
./fkpt chatty=1 Nk=60 nquadSteps=300

# Double resolution (should be ~2× slower)
./fkpt chatty=1 Nk=240 nquadSteps=300
```

### Vary integration quadrature steps (should scale linearly):

```bash
# Half steps (should be ~2× faster, less accurate)
./fkpt chatty=1 Nk=120 nquadSteps=150

# Double steps (should be ~2× slower, more accurate)
./fkpt chatty=1 Nk=120 nquadSteps=600
```

### Expected scaling:

- **Time ∝ Nk**: Linear scaling with k-grid points
- **Time ∝ nquadSteps**: Linear scaling with outer integration loop
- **Time ∝ Nk × nquadSteps**: Total complexity

Default: Nk=120, nquadSteps=300 → ~36,000 k-function evaluations

## Saving Timing Results

Redirect output to file for later analysis:

```bash
./fkpt chatty=1 ... > timing_output.txt 2>&1
```

Then extract timing information:

```bash
# Get MainLoop breakdown
grep -A 7 "TIMING BREAKDOWN" timing_output.txt

# Get compute_kfunctions breakdown
grep -A 6 "compute_kfunctions breakdown" timing_output.txt

# Get per-k timings
grep "k\[.*\]:" timing_output.txt > per_k_times.dat
```

## Interpreting Results

### Scenario 1: compute_kfunctions dominates (>90%)

**Expected behavior.** This confirms our hypothesis from `compute.md`.

**Next steps:**
- Profile with gprof or Instruments (Approach 2 or 3 from PROFILING_PLAN.md)
- Investigate optimization opportunities in `ki_functions()`

### Scenario 2: compute_rsdmultipoles is significant (>10%)

**Unexpected.** The RSD multipoles should be fast since they use pre-computed k-functions.

**Possible causes:**
- Large Nk with many angular quadrature points
- I/O bottleneck in reading k-functions
- Check NGL parameter in `src/rsd.c` (default: 16)

### Scenario 3: Initialization or I/O dominates (>5%)

**Unexpected.** Suggests I/O or memory allocation issues.

**Possible causes:**
- Large input linear power spectrum file
- Slow disk I/O
- Memory allocation overhead (check golists.Nk)

## Limitations of This Approach

1. **No function-level detail**: Can't see time in specific functions like `splint()` or `Interpolation_nr()`
2. **No cache analysis**: Can't measure cache misses or memory bandwidth
3. **Manual instrumentation**: Need to recompile after any changes
4. **Timing overhead**: Small but non-zero overhead from `second()` calls

**Solution**: Use gprof, Instruments, or Callgrind for deeper profiling (see PROFILING_PLAN.md Approaches 2-5).

## Comparison with Other Approaches

| Approach | Pros | Cons | When to Use |
|----------|------|------|-------------|
| **Manual (this)** | Simple, portable, minimal overhead | Limited detail, requires code changes | Initial assessment, validate hypotheses |
| **gprof** | Function-level detail, call graph | Sampling (may miss short functions) | After confirming hotspots |
| **Instruments** | Line-level detail, GUI, flamegraph | macOS only, learning curve | Deep dive into specific functions |
| **Callgrind** | Instruction counts, deterministic | Very slow (10-50× overhead) | Final optimization, cache analysis |

## Next Steps

1. **Run with default parameters** to establish baseline
2. **Confirm compute_kfunctions dominates** (>90%)
3. **Test scaling** with Nk and nquadSteps
4. **Deep profiling** with gprof/Instruments if needed
5. **Identify optimization targets** based on results

See `PROFILING_PLAN.md` for the complete profiling workflow and next steps.

## Reverting to Original Code

To return to the main branch without timing instrumentation:

```bash
git checkout main
make clean
make
```

To return to this trace branch:

```bash
git checkout trace
make clean
make
```
