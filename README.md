# Go-MicroGPT

Tiny GPT-like character model in pure Go, inspired by the microgpt/makemore style.
Training and interfacing GPTs using pure, dependency free Golang.

## What this project does

- Trains a small decoder-only model on names.
- Uses a lightweight autograd engine implemented in Go.
- Generates new ("hallucinated") names after training.
- Supports native CLI execution and browser execution through WASM.

## Run (native)

```bash
go run .
```

If `input.txt` is missing in native mode, it is downloaded automatically from the names dataset URL in `main.go`.

## Run in browser (WASM)

Build wasm artifacts:

```bash
scripts/build_wasm.sh
```

Serve static files:

```bash
cd web && python3 -m http.server 8080
```

Open:

```text
http://localhost:8080
```

Notes:
- `web/index.html` is the loader UI.
- Browser mode uses a built-in fallback mini dataset (WASM cannot read local files directly like native Go).

## Profiling support

You can generate profiles using env vars:

```bash
MICROGPT_CPU_PROFILE=cpu.pprof MICROGPT_MEM_PROFILE=mem.pprof go run .
```

Inspect:

```bash
go tool pprof -top ./go_microgpt cpu.pprof
go tool pprof -top -alloc_space ./go_microgpt mem.pprof
```

## Optimization summary

The codebase was optimized in multiple focused passes, keeping behavior the same while reducing allocations and runtime overhead.

### Major optimizations applied

1. **Adaptive concurrency in GPT forward**
   - Parallelized Q/K/V and attention heads only when work is large enough.
   - Avoided goroutine overhead on tiny workloads.

2. **Reduced attention allocations**
   - Removed temporary per-head key/value slice construction.
   - Indexed directly into cached layer vectors.

3. **Precomputed layer parameter keys**
   - Removed repeated `fmt.Sprintf` calls on hot token loops.

4. **Inference-only numeric fast path**
   - Added no-autograd inference (`float64`) for sampling.
   - Avoided building autograd graphs during generation.

5. **Autograd graph memory optimization**
   - Added `sync.Pool` for temporary `Value` nodes.
   - Replaced per-backward visited map with mark-based traversal.
   - Released graph nodes after each step.

6. **Tensor-style cross-entropy head during training**
   - Replaced autograd `Softmax->Log->Neg` chain with numeric CE and direct logits gradients (`probs - onehot`).

7. **Fused autograd primitives**
   - Added `Dot(...)` and `WeightedSum(...)` ops to collapse many `Add/Mul` nodes into single nodes.

8. **Pooled buffers for fused ops**
   - Reused children/local-grad slices for common small sizes (8/16/32).

9. **Step-level scratch reuse**
   - Reused keys/values/loss buffers/logit scratch across train and inference loops.

10. **Fused RMSNorm autograd**
    - Implemented analytic RMSNorm local gradients in a single custom op path.

11. **Range-over-int cleanup**
    - Updated counted loops to `for i := range n` style on Go 1.25.

## Measured runtime comparison

Measurements below are from repeated end-to-end `go run .` runs on the same machine during optimization work.
They are approximate and include run-to-run noise, but show the trend clearly.

| Stage | Elapsed |
|---|---:|
| Early baseline | 23.40s |
| Adaptive concurrency + key/allocation cleanup | 22.27s |
| Numeric inference path | 21.71s |
| Graph pooling + mark traversal | 15.87s |
| Numeric CE training head | 14.98s |
| Fused Dot/WeightedSum ops | 3.17s |
| Buffer pooling + scratch reuse + fused RMSNorm | 2.97s |

Overall speedup from the recorded baseline: **~7.9x** (`23.40s -> 2.97s`).

## Remaining improvement ideas

- Tune `GOGC` (`200`/`300`) for this allocation profile.
- Add microbenchmarks (`go test -bench`) for key kernels.
- For bigger models: move to contiguous tensor core and/or BLAS/GPU backend.
