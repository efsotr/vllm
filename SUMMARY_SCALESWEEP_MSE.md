# ScaleSweep MSE NVFP4 Summary

This file tracks every code and test change for the `scalesweep_mse` and
`scalesweep_mse128` NVFP4 activation quantization paths.

## Launch command comparison

Replace `NVFP4_MODEL_ID_OR_PATH` with the same NVFP4 model ID or local path.
The commands fix the GEMM backend to CUTLASS so only activation quantization
changes between runs.

Standard NVFP4:

```bash
vllm serve NVFP4_MODEL_ID_OR_PATH --linear-backend cutlass
```

ScaleSweep around the block's base scale:

```bash
vllm serve NVFP4_MODEL_ID_OR_PATH --linear-backend cutlass \
  --act-quant-backend scalesweep_mse
```

ScaleSweep over all positive finite FP8 E4M3 scales:

```bash
vllm serve NVFP4_MODEL_ID_OR_PATH --linear-backend cutlass \
  --act-quant-backend scalesweep_mse128
```

These options change only dense NVFP4 activation quantization. They do not
change checkpoint weight quantization or the CUTLASS GEMM backend. FlashInfer
NVFP4 kernels keep their original activation quantizers.

## Change trace

### Quantizers and dispatch

- [`scalesweep_mse_nvfp4_utils.py`](vllm/model_executor/layers/quantization/utils/scalesweep_mse_nvfp4_utils.py)
  implements the shared Triton quantizer and registers default and `.out`
  custom ops for both modes. `scalesweep_mse` checks FP8 raw scales from
  `base_raw - 3` through `base_raw + 7`; `scalesweep_mse128` enumerates raw
  values `1..126` and bitcasts each `uint8` value to FP8 E4M3. Both modes keep
  the scale with the lowest FP4 E2M1 reconstruction error and support linear,
  128x4-swizzled, and padded outputs.
- [`vllm/_custom_ops.py`](vllm/_custom_ops.py) dispatches
  `scaled_fp4_quant` to the matching ScaleSweep `.out` op for
  `backend="scalesweep_mse"` or `backend="scalesweep_mse128"`.

### Configuration and NVFP4 linear integration

- [`vllm/config/kernel.py`](vllm/config/kernel.py) defines both ScaleSweep
  activation backend values and normalizes CLI spelling.
- [`vllm/engine/arg_utils.py`](vllm/engine/arg_utils.py) exposes
  `--act-quant-backend` and transfers it into `KernelConfig`.
- [`vllm/model_executor/kernels/linear/__init__.py`](vllm/model_executor/kernels/linear/__init__.py)
  reads the configured activation backend during NVFP4 kernel initialization.
- [`vllm/model_executor/kernels/linear/nvfp4/base.py`](vllm/model_executor/kernels/linear/nvfp4/base.py)
  defines the accepted backend type and stores the override in the layer
  configuration.
- [`vllm/model_executor/kernels/linear/nvfp4/__init__.py`](vllm/model_executor/kernels/linear/nvfp4/__init__.py)
  exports the activation backend type.
- [`vllm/model_executor/kernels/linear/nvfp4/cutlass.py`](vllm/model_executor/kernels/linear/nvfp4/cutlass.py)
  applies the configured activation quantizer to the CUTLASS NVFP4 path.
- [`vllm/model_executor/kernels/linear/nvfp4/flashinfer.py`](vllm/model_executor/kernels/linear/nvfp4/flashinfer.py)
  is restored to its original behavior and does not apply the ScaleSweep
  override.

### Benchmark and tests

- [`benchmark_nvfp4_quant.py`](benchmarks/kernels/benchmark_nvfp4_quant.py) adds
  linear and swizzled providers for both ScaleSweep modes.
- [`test_scalesweep_mse_nvfp4_quant.py`](tests/kernels/quantization/test_scalesweep_mse_nvfp4_quant.py)
  covers FP16/BF16 correctness, custom-op dispatch, scale layouts, padding,
  bounded candidates, and the full `1..126` reference sweep.
- [`SUMMARY_SCALESWEEP_MSE.md`](SUMMARY_SCALESWEEP_MSE.md) records the command
  comparison and this complete file-level trace.
