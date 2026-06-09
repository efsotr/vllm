# ScaleSweep MSE NVFP4 Summary

This file is the required implementation summary for the `scalesweep_mse`
NVFP4 activation quantization path.

Any change that modifies `scalesweep_mse` behavior, interfaces, supported
features, registration, configuration, or tests must update this file in the
same change.

## Public Interfaces

`vllm._custom_ops.scaled_fp4_quant` supports `backend="scalesweep_mse"`.
When selected, it allocates FP4 output tensors through
`create_fp4_output_tensors` and calls the ScaleSweep MSE `.out` custom op.

The direct custom ops live in:

`vllm/model_executor/layers/quantization/utils/scalesweep_mse_nvfp4_utils.py`

Registered ops:

- `torch.ops.vllm.scalesweep_mse_nvfp4_quant`
- `torch.ops.vllm.scalesweep_mse_nvfp4_quant.out`

The argument naming follows `torch.ops._C.scaled_fp4_quant`:

- default: `input`, `input_scale`, `is_sf_swizzled_layout`, `padded_n`
- out: `input`, `input_scale`, `is_sf_swizzled_layout`, `output`,
  `output_scale`

The out variant intentionally does not take `padded_n`; it infers the physical
output K dimension from `output.shape[-1] * 2`.

## Quantization Algorithm

ScaleSweep MSE quantizes each 16-element block independently.

For each block:

1. Load 16 input values and multiply by `input_scale`.
2. Compute `abs_max / FP4_E2M1_MAX` as the base scale.
3. Convert the base scale to FP8 E4M3 raw bits.
4. Sweep candidate FP8 scale raw values from
   `base_raw + LOWER_BOUND` through `base_raw + UPPER_BOUND`.
5. Clamp candidate raw values to `[1, REF_MAX_SCALE_RAW]`.
6. Quantize the scaled values to FP4 E2M1 and compute reconstruction MSE.
7. Store the candidate with the lowest MSE.

Constants:

- `BLOCK_SIZE = 16`
- `LOWER_BOUND = -3`
- `UPPER_BOUND = 7`
- `FP4_E2M1_MAX = 6.0`
- `REF_MAX_SCALE_RAW = 126`

## Padded Output Behavior

The kernel launches over the output block space:

`num_output_blocks = num_row * blocks_per_col_out`

Input and output column block counts are separate:

- `BLOCKS_PER_COL_IN = input.shape[-1] // 16`
- `BLOCKS_PER_COL_OUT = (output.shape[-1] * 2) // 16`

For padded output blocks where `col >= BLOCKS_PER_COL_IN`, the input load mask
is false. Loads therefore produce zeros, while stores still write the output
block. This yields:

- FP4 padded output bytes equal to `0`
- padded scale raw bytes equal to `1`, the minimum positive FP8 E4M3 value

The implementation must not pre-clear `output` or `output_scale` for padded
output; padding is produced by the kernel itself.

## Scale Layouts

ScaleSweep supports both scale layouts used by `scaled_fp4_quant`:

- `is_sf_swizzled_layout=True`: swizzled 128x4 scale layout
- `is_sf_swizzled_layout=False`: linear per-block scale layout

For swizzled output scales, offsets are computed with the output column block
count and the padded output column block count. For non-swizzled output scales,
the output block offset is used directly.

## Kernel Specialization

The Triton kernel uses autotune configs keyed by:

- `LOG2_NUM_ROW`
- `BLOCKS_PER_COL_IN`
- `BLOCKS_PER_COL_OUT`
- `LOWER_BOUND`
- `NUM_CANDIDATES`
- `MAX_SCALE_RAW`
- `IS_SWIZZLE_SCALE`
- `BLOCKS_PER_COL_OUT_PAD`

`LOG2_NUM_ROW` is a heuristic computed as `int(math.log2(NUM_ROW))`.

## Linear Kernel Integration

`KernelConfig` exposes `act_quant_backend`.

Supported values include:

- `auto`
- `cutlass`
- `flashinfer_cutlass`
- `flashinfer_trtllm`
- `flashinfer_cudnn`
- `b12x`
- `fbgemm`
- `scalesweep_mse`

`auto` preserves the activation quantization backend selected by the NVFP4 GEMM
backend. Setting `act_quant_backend="scalesweep_mse"` routes CUTLASS and
FlashInfer NVFP4 activation quantization through ScaleSweep MSE while keeping
the selected GEMM backend unchanged.

The selected value is stored in `NvFp4LinearLayerConfig` and passed through
`init_nvfp4_linear_kernel`.

## Emulation Mode

Setting `SCALESWEEP_MSE_EMULATION=1` switches only the ScaleSweep MSE FP4
conversion pieces from inline asm to Triton emulation:

- FP32-to-FP4 packing for the final output bytes
- FP32-to-FP4 rounding used during candidate MSE evaluation

All other ScaleSweep MSE kernel logic, including FP8 conversion, scale sweep,
scale layout handling, padding behavior, and stores, remains unchanged.

When the NVFP4 linear backend is `emulation`, the same environment variable
switches activation handling to:

1. `scaled_fp4_quant(..., backend="scalesweep_mse")`
2. dequantize the activation FP4 result to BF16
3. dequantize NVFP4 weights to BF16
4. call `torch.nn.functional.linear`

Without `SCALESWEEP_MSE_EMULATION=1`, the existing NVFP4 emulation path is
unchanged.

## Benchmarks

Benchmark coverage lives in:

`benchmarks/kernels/benchmark_nvfp4_quant.py`

The NVFP4 input quantization benchmark includes ScaleSweep MSE providers:

- `scalesweep_mse`: non-swizzled scale layout
- `scalesweep_mse-swizzle`: swizzled 128x4 scale layout

These run through `ops.scaled_fp4_quant(..., backend="scalesweep_mse")` and are
reported alongside the existing vLLM and FlashInfer quantization providers.

## Tests

ScaleSweep MSE tests live in:

`tests/kernels/quantization/test_scalesweep_mse_nvfp4_quant.py`

Test coverage:

- `test_scalesweep_mse_nvfp4_quant` - verifies
  `ops.scaled_fp4_quant(..., backend="scalesweep_mse")` against the MSE
  reference for FP16 and BF16 inputs, regular output shapes, and both swizzled
  and non-swizzled scale layouts.
- `test_scalesweep_mse_nvfp4_quant_default_and_out` - verifies direct default
  custom op registration and direct `.out` custom op registration against the
  MSE reference for regular output shapes and both scale layouts.
- `test_scalesweep_mse_nvfp4_quant_padded_output` - verifies padded output
  shapes against the MSE reference for both scale layouts, including padded FP4
  output bytes equal to zero and padded scale raw bytes equal to `1`.
- `test_scalesweep_mse_nvfp4_quant_bound_examples` - verifies known lower-bound
  and upper-bound scale sweep examples against the MSE reference for FP16 and
  BF16 inputs and both scale layouts.
- `test_scalesweep_mse_matches_all_positive_fp8_reference` - verifies that the
  bounded ScaleSweep MSE reference chooses the same FP4 values and FP8 scales as
  a separate `torch.compile` reference that enumerates every positive FP8 E4M3
  raw scale from `1` through `REF_MAX_SCALE_RAW`.
- `test_scalesweep_mse_nvfp4_quant_triton_emulation` - verifies
  `SCALESWEEP_MSE_EMULATION=1`, covering the Triton-emulated FP4 conversion
  path for both scale layouts.

The reference implementation computes `input_scale` as:

`256.0 * scalar_types.float4_e2m1f.max()` divided by `abs(input).max()`.

This matches the reference behavior used by the standalone ScaleSweep MSE
prototype.
