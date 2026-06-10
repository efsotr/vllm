# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os

import torch

from vllm._custom_ops import create_fp4_output_tensors
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import direct_register_custom_op

BLOCK_SIZE = 16
LOWER_BOUND = -3
UPPER_BOUND = 7
FP4_E2M1_MAX = 6.0
REF_MAX_SCALE_RAW = 126


def _use_scalesweep_mse_emulation() -> bool:
    return bool(int(os.getenv("SCALESWEEP_MSE_EMULATION", "0")))


@triton.jit
def _swizzled_scale_offsets(
    row, col,
    BLOCKS_PER_COL_OUT_PAD: tl.constexpr,
):
    major_m = row >> 7
    row_in_tile = row & 127
    tile_m = row_in_tile >> 5
    inner_m = row & 31

    major_k = col >> 2
    inner_k = col & 3

    return (
        major_m * (BLOCKS_PER_COL_OUT_PAD * 128)
        + major_k * 512
        + inner_m * 16
        + tile_m * 4
        + inner_k
    )


@triton.jit
def _fp32x16_to_e2m1_u32x2(
    x0, x1, x2, x3,
    x4, x5, x6, x7,
    x8, x9, x10, x11,
    x12, x13, x14, x15,
):
    lo, hi = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .b8 b0;
          .reg .b8 b1;
          .reg .b8 b2;
          .reg .b8 b3;
          .reg .b8 b4;
          .reg .b8 b5;
          .reg .b8 b6;
          .reg .b8 b7;

          cvt.rn.satfinite.e2m1x2.f32 b0,  $3,  $2;
          cvt.rn.satfinite.e2m1x2.f32 b1,  $5,  $4;
          cvt.rn.satfinite.e2m1x2.f32 b2,  $7,  $6;
          cvt.rn.satfinite.e2m1x2.f32 b3,  $9,  $8;
          cvt.rn.satfinite.e2m1x2.f32 b4,  $11, $10;
          cvt.rn.satfinite.e2m1x2.f32 b5,  $13, $12;
          cvt.rn.satfinite.e2m1x2.f32 b6,  $15, $14;
          cvt.rn.satfinite.e2m1x2.f32 b7,  $17, $16;

          mov.b32 $0, {b0, b1, b2, b3};
          mov.b32 $1, {b4, b5, b6, b7};
        }
        """,
        constraints="=r,=r,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f",
        args=[
            x0, x1, x2, x3,
            x4, x5, x6, x7,
            x8, x9, x10, x11,
            x12, x13, x14, x15,
        ],
        dtype=(tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )
    return lo, hi


@triton.jit
def _fp32_to_e2m1_nibble_emulation(x):
    sign_bit = tl.where(x < 0.0, 8, 0).to(tl.uint32)
    abs_x = tl.abs(x)
    magnitude = tl.full(abs_x.shape, 0, tl.uint32)
    magnitude = tl.where((abs_x > 0.25) & (abs_x < 0.75), 1, magnitude)
    magnitude = tl.where((abs_x >= 0.75) & (abs_x <= 1.25), 2, magnitude)
    magnitude = tl.where((abs_x > 1.25) & (abs_x < 1.75), 3, magnitude)
    magnitude = tl.where((abs_x >= 1.75) & (abs_x <= 2.5), 4, magnitude)
    magnitude = tl.where((abs_x > 2.5) & (abs_x < 3.5), 5, magnitude)
    magnitude = tl.where((abs_x >= 3.5) & (abs_x <= 5.0), 6, magnitude)
    magnitude = tl.where(abs_x > 5.0, 7, magnitude)
    return sign_bit | magnitude


@triton.jit
def _fp32_to_e2m1_float_emulation(x):
    sign = tl.where(x < 0.0, -1.0, 1.0)
    abs_x = tl.abs(x)
    out = tl.full(abs_x.shape, 0.0, tl.float32)
    out = tl.where((abs_x > 0.25) & (abs_x < 0.75), 0.5, out)
    out = tl.where((abs_x >= 0.75) & (abs_x <= 1.25), 1.0, out)
    out = tl.where((abs_x > 1.25) & (abs_x < 1.75), 1.5, out)
    out = tl.where((abs_x >= 1.75) & (abs_x <= 2.5), 2.0, out)
    out = tl.where((abs_x > 2.5) & (abs_x < 3.5), 3.0, out)
    out = tl.where((abs_x >= 3.5) & (abs_x <= 5.0), 4.0, out)
    out = tl.where(abs_x > 5.0, 6.0, out)
    return out * sign


@triton.jit
def _fp32x16_to_e2m1_u32x2_emulation(
    x0, x1, x2, x3,
    x4, x5, x6, x7,
    x8, x9, x10, x11,
    x12, x13, x14, x15,
):
    b0 = _fp32_to_e2m1_nibble_emulation(x0) | (
        _fp32_to_e2m1_nibble_emulation(x1) << 4
    )
    b1 = _fp32_to_e2m1_nibble_emulation(x2) | (
        _fp32_to_e2m1_nibble_emulation(x3) << 4
    )
    b2 = _fp32_to_e2m1_nibble_emulation(x4) | (
        _fp32_to_e2m1_nibble_emulation(x5) << 4
    )
    b3 = _fp32_to_e2m1_nibble_emulation(x6) | (
        _fp32_to_e2m1_nibble_emulation(x7) << 4
    )
    b4 = _fp32_to_e2m1_nibble_emulation(x8) | (
        _fp32_to_e2m1_nibble_emulation(x9) << 4
    )
    b5 = _fp32_to_e2m1_nibble_emulation(x10) | (
        _fp32_to_e2m1_nibble_emulation(x11) << 4
    )
    b6 = _fp32_to_e2m1_nibble_emulation(x12) | (
        _fp32_to_e2m1_nibble_emulation(x13) << 4
    )
    b7 = _fp32_to_e2m1_nibble_emulation(x14) | (
        _fp32_to_e2m1_nibble_emulation(x15) << 4
    )

    lo = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    hi = b4 | (b5 << 8) | (b6 << 16) | (b7 << 24)
    return lo, hi


@triton.jit
def _fp32x2_e2m1_quant_squared_error(x0, x1):
    return tl.inline_asm_elementwise(
        asm=r"""
        {
          .reg .b8  b;
          .reg .b32 h;
          .reg .b16 lo;
          .reg .b16 hi;
          .reg .f32 q0;
          .reg .f32 q1;
          .reg .f32 d0;
          .reg .f32 d1;

          cvt.rn.satfinite.e2m1x2.f32 b, $2, $1;
          cvt.rn.f16x2.e2m1x2 h, b;

          mov.b32 {lo, hi}, h;
          cvt.f32.f16 q0, lo;
          cvt.f32.f16 q1, hi;

          sub.rn.f32 d0, q0, $1;
          sub.rn.f32 d1, q1, $2;
          mul.rn.f32 d0, d0, d0;
          fma.rn.f32 $0, d1, d1, d0;
        }
        """,
        constraints="=f,f,f",
        args=[x0, x1],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp32x2_e2m1_quant_squared_error_acc(acc, x0, x1):
    return tl.inline_asm_elementwise(
        asm=r"""
        {
          .reg .b8  b;
          .reg .b32 h;
          .reg .b16 lo;
          .reg .b16 hi;
          .reg .f32 q0;
          .reg .f32 q1;
          .reg .f32 d0;
          .reg .f32 d1;

          cvt.rn.satfinite.e2m1x2.f32 b, $2, $1;
          cvt.rn.f16x2.e2m1x2 h, b;

          mov.b32 {lo, hi}, h;
          cvt.f32.f16 q0, lo;
          cvt.f32.f16 q1, hi;

          sub.rn.f32 d0, q0, $1;
          sub.rn.f32 d1, q1, $2;
          fma.rn.f32 d0, d0, d0, $3;
          fma.rn.f32 $0, d1, d1, d0;
        }
        """,
        constraints="=f,f,f,f",
        args=[x0, x1, acc],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _scaled_fp32x16_e2m1_quant_squared_error(
    x0, x1, x2, x3,
    x4, x5, x6, x7,
    x8, x9, x10, x11,
    x12, x13, x14, x15,
):
    err = _fp32x2_e2m1_quant_squared_error(x0, x1)
    err = _fp32x2_e2m1_quant_squared_error_acc(err, x2, x3)
    err = _fp32x2_e2m1_quant_squared_error_acc(err, x4, x5)
    err = _fp32x2_e2m1_quant_squared_error_acc(err, x6, x7)
    err = _fp32x2_e2m1_quant_squared_error_acc(err, x8, x9)
    err = _fp32x2_e2m1_quant_squared_error_acc(err, x10, x11)
    err = _fp32x2_e2m1_quant_squared_error_acc(err, x12, x13)
    return _fp32x2_e2m1_quant_squared_error_acc(err, x14, x15)


@triton.jit
def _scaled_fp32x16_e2m1_quant_squared_error_emulation(
    x0, x1, x2, x3,
    x4, x5, x6, x7,
    x8, x9, x10, x11,
    x12, x13, x14, x15,
):
    q0 = _fp32_to_e2m1_float_emulation(x0)
    q1 = _fp32_to_e2m1_float_emulation(x1)
    q2 = _fp32_to_e2m1_float_emulation(x2)
    q3 = _fp32_to_e2m1_float_emulation(x3)
    q4 = _fp32_to_e2m1_float_emulation(x4)
    q5 = _fp32_to_e2m1_float_emulation(x5)
    q6 = _fp32_to_e2m1_float_emulation(x6)
    q7 = _fp32_to_e2m1_float_emulation(x7)
    q8 = _fp32_to_e2m1_float_emulation(x8)
    q9 = _fp32_to_e2m1_float_emulation(x9)
    q10 = _fp32_to_e2m1_float_emulation(x10)
    q11 = _fp32_to_e2m1_float_emulation(x11)
    q12 = _fp32_to_e2m1_float_emulation(x12)
    q13 = _fp32_to_e2m1_float_emulation(x13)
    q14 = _fp32_to_e2m1_float_emulation(x14)
    q15 = _fp32_to_e2m1_float_emulation(x15)

    err = (q0 - x0) * (q0 - x0)
    err += (q1 - x1) * (q1 - x1)
    err += (q2 - x2) * (q2 - x2)
    err += (q3 - x3) * (q3 - x3)
    err += (q4 - x4) * (q4 - x4)
    err += (q5 - x5) * (q5 - x5)
    err += (q6 - x6) * (q6 - x6)
    err += (q7 - x7) * (q7 - x7)
    err += (q8 - x8) * (q8 - x8)
    err += (q9 - x9) * (q9 - x9)
    err += (q10 - x10) * (q10 - x10)
    err += (q11 - x11) * (q11 - x11)
    err += (q12 - x12) * (q12 - x12)
    err += (q13 - x13) * (q13 - x13)
    err += (q14 - x14) * (q14 - x14)
    err += (q15 - x15) * (q15 - x15)
    return err


if _use_scalesweep_mse_emulation():
    _fp32x16_to_e2m1_u32x2 = _fp32x16_to_e2m1_u32x2_emulation
    _scaled_fp32x16_e2m1_quant_squared_error = (
        _scaled_fp32x16_e2m1_quant_squared_error_emulation
    )


@triton.jit
def _fp32x16_e2m1_quant_squared_error(
    v0, v1, v2, v3,
    v4, v5, v6, v7,
    v8, v9, v10, v11,
    v12, v13, v14, v15,
    inv_scale,
    scale,
):
    squared_error = _scaled_fp32x16_e2m1_quant_squared_error(
        v0 * inv_scale, v1 * inv_scale,
        v2 * inv_scale, v3 * inv_scale,
        v4 * inv_scale, v5 * inv_scale,
        v6 * inv_scale, v7 * inv_scale,
        v8 * inv_scale, v9 * inv_scale,
        v10 * inv_scale, v11 * inv_scale,
        v12 * inv_scale, v13 * inv_scale,
        v14 * inv_scale, v15 * inv_scale,
    )
    return squared_error * (scale * scale)


@triton.jit
def _max_abs_16(
    v0, v1, v2, v3,
    v4, v5, v6, v7,
    v8, v9, v10, v11,
    v12, v13, v14, v15,
):
    m0 = tl.maximum(tl.abs(v0), tl.abs(v1))
    m1 = tl.maximum(tl.abs(v2), tl.abs(v3))
    m2 = tl.maximum(tl.abs(v4), tl.abs(v5))
    m3 = tl.maximum(tl.abs(v6), tl.abs(v7))
    m4 = tl.maximum(tl.abs(v8), tl.abs(v9))
    m5 = tl.maximum(tl.abs(v10), tl.abs(v11))
    m6 = tl.maximum(tl.abs(v12), tl.abs(v13))
    m7 = tl.maximum(tl.abs(v14), tl.abs(v15))

    m01 = tl.maximum(m0, m1)
    m23 = tl.maximum(m2, m3)
    m45 = tl.maximum(m4, m5)
    m67 = tl.maximum(m6, m7)
    return tl.maximum(tl.maximum(m01, m23), tl.maximum(m45, m67))


@triton.jit
def _load_normalized_16_cols(ptr, block_offsets, block_mask, global_scale_inv):
    base_elem = block_offsets * 16

    v0 = tl.load(ptr + base_elem + 0, mask=block_mask, other=0.0).to(tl.float32)
    v1 = tl.load(ptr + base_elem + 1, mask=block_mask, other=0.0).to(tl.float32)
    v2 = tl.load(ptr + base_elem + 2, mask=block_mask, other=0.0).to(tl.float32)
    v3 = tl.load(ptr + base_elem + 3, mask=block_mask, other=0.0).to(tl.float32)
    v4 = tl.load(ptr + base_elem + 4, mask=block_mask, other=0.0).to(tl.float32)
    v5 = tl.load(ptr + base_elem + 5, mask=block_mask, other=0.0).to(tl.float32)
    v6 = tl.load(ptr + base_elem + 6, mask=block_mask, other=0.0).to(tl.float32)
    v7 = tl.load(ptr + base_elem + 7, mask=block_mask, other=0.0).to(tl.float32)
    v8 = tl.load(ptr + base_elem + 8, mask=block_mask, other=0.0).to(tl.float32)
    v9 = tl.load(ptr + base_elem + 9, mask=block_mask, other=0.0).to(tl.float32)
    v10 = tl.load(ptr + base_elem + 10, mask=block_mask, other=0.0).to(tl.float32)
    v11 = tl.load(ptr + base_elem + 11, mask=block_mask, other=0.0).to(tl.float32)
    v12 = tl.load(ptr + base_elem + 12, mask=block_mask, other=0.0).to(tl.float32)
    v13 = tl.load(ptr + base_elem + 13, mask=block_mask, other=0.0).to(tl.float32)
    v14 = tl.load(ptr + base_elem + 14, mask=block_mask, other=0.0).to(tl.float32)
    v15 = tl.load(ptr + base_elem + 15, mask=block_mask, other=0.0).to(tl.float32)

    return (
        v0 * global_scale_inv, v1 * global_scale_inv,
        v2 * global_scale_inv, v3 * global_scale_inv,
        v4 * global_scale_inv, v5 * global_scale_inv,
        v6 * global_scale_inv, v7 * global_scale_inv,
        v8 * global_scale_inv, v9 * global_scale_inv,
        v10 * global_scale_inv, v11 * global_scale_inv,
        v12 * global_scale_inv, v13 * global_scale_inv,
        v14 * global_scale_inv, v15 * global_scale_inv,
    )


SCALESWEEP_CONFIGS = [
    triton.Config({"BLOCKS_PER_PROGRAM": 32}, num_warps=1),
    triton.Config({"BLOCKS_PER_PROGRAM": 64}, num_warps=2),
    triton.Config({"BLOCKS_PER_PROGRAM": 128}, num_warps=4),
    triton.Config({"BLOCKS_PER_PROGRAM": 256}, num_warps=8),
    triton.Config({"BLOCKS_PER_PROGRAM": 512}, num_warps=16),
    triton.Config({"BLOCKS_PER_PROGRAM": 1024}, num_warps=32),
]


@triton.heuristics({"LOG2_NUM_ROW": lambda args: int(math.log2(args["NUM_ROW"]))})
@triton.autotune(
    configs=SCALESWEEP_CONFIGS,
    key=[
        "LOG2_NUM_ROW",
        "BLOCKS_PER_COL_IN",
        "BLOCKS_PER_COL_OUT",
        "LOWER_BOUND",
        "NUM_CANDIDATES",
        "MAX_SCALE_RAW",
        "IS_SWIZZLE_SCALE",
        "BLOCKS_PER_COL_OUT_PAD",
    ],
)
@triton.jit
def _scalesweep_mse_nvfp4_quant_kernel(
    input_ptr,
    output_scale_ptr,
    output_i32_ptr,
    global_scale_inv_ptr,
    NUM_OUTPUT_BLOCKS: tl.constexpr,
    NUM_ROW: tl.constexpr,
    BLOCKS_PER_COL_IN: tl.constexpr,
    BLOCKS_PER_COL_OUT: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    MAX_SCALE_RAW: tl.constexpr,
    IS_SWIZZLE_SCALE: tl.constexpr,
    BLOCKS_PER_COL_OUT_PAD: tl.constexpr,
    LOG2_NUM_ROW: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    global_scale_inv = tl.load(global_scale_inv_ptr)
    pid = tl.program_id(0)
    output_block_offsets = pid * BLOCKS_PER_PROGRAM + tl.arange(0, BLOCKS_PER_PROGRAM)
    output_block_mask = output_block_offsets < NUM_OUTPUT_BLOCKS

    row = output_block_offsets // BLOCKS_PER_COL_OUT
    col = output_block_offsets % BLOCKS_PER_COL_OUT
    if BLOCKS_PER_COL_IN == BLOCKS_PER_COL_OUT:
        input_block_offsets = output_block_offsets
        input_block_mask = output_block_mask
    else:
        input_block_offsets = row * BLOCKS_PER_COL_IN + col
        input_block_mask = output_block_mask & (col < BLOCKS_PER_COL_IN)

    (
        v0, v1, v2, v3,
        v4, v5, v6, v7,
        v8, v9, v10, v11,
        v12, v13, v14, v15,
    ) = _load_normalized_16_cols(
        input_ptr,
        input_block_offsets,
        input_block_mask,
        global_scale_inv,
    )

    abs_max = _max_abs_16(
        v0, v1, v2, v3,
        v4, v5, v6, v7,
        v8, v9, v10, v11,
        v12, v13, v14, v15,
    )
    base_scale = abs_max * (1.0 / 6.0)
    base_raw = base_scale.to(tl.float8e4nv).to(tl.uint8, bitcast=True).to(tl.int32)

    best_mse = tl.full((BLOCKS_PER_PROGRAM,), float("inf"), tl.float32)
    best_scale_fp8 = tl.full((BLOCKS_PER_PROGRAM,), 0, tl.float8e4nv)

    for i in tl.static_range(0, NUM_CANDIDATES):
        raw_i = tl.minimum(
            tl.maximum(base_raw + (LOWER_BOUND + i), 1),
            MAX_SCALE_RAW,
        ).to(tl.uint8)
        scale_fp8 = raw_i.to(tl.float8e4nv, bitcast=True)
        scale_i = scale_fp8.to(tl.float32)
        inv_scale_i = 1.0 / scale_i

        mse_i = _fp32x16_e2m1_quant_squared_error(
            v0, v1, v2, v3,
            v4, v5, v6, v7,
            v8, v9, v10, v11,
            v12, v13, v14, v15,
            inv_scale_i,
            scale_i,
        )

        better = mse_i < best_mse
        best_mse = tl.where(better, mse_i, best_mse)
        best_scale_fp8 = tl.where(better, scale_fp8, best_scale_fp8)

    if IS_SWIZZLE_SCALE:
        scale_offsets = _swizzled_scale_offsets(
            row,
            col,
            BLOCKS_PER_COL_OUT_PAD,
        )
    else:
        scale_offsets = output_block_offsets

    tl.store(
        output_scale_ptr + scale_offsets,
        best_scale_fp8,
        mask=output_block_mask,
    )

    inv_scale = 1.0 / best_scale_fp8.to(tl.float32)
    lo, hi = _fp32x16_to_e2m1_u32x2(
        v0 * inv_scale, v1 * inv_scale,
        v2 * inv_scale, v3 * inv_scale,
        v4 * inv_scale, v5 * inv_scale,
        v6 * inv_scale, v7 * inv_scale,
        v8 * inv_scale, v9 * inv_scale,
        v10 * inv_scale, v11 * inv_scale,
        v12 * inv_scale, v13 * inv_scale,
        v14 * inv_scale, v15 * inv_scale,
    )

    output_i32_offsets = output_block_offsets * 2
    tl.store(output_i32_ptr + output_i32_offsets, lo, mask=output_block_mask)
    tl.store(output_i32_ptr + output_i32_offsets + 1, hi, mask=output_block_mask)


def scalesweep_mse_nvfp4_quant_out(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    *,
    output: torch.Tensor,
    output_scale: torch.Tensor,
) -> None:
    num_row, n = input.shape
    physical_n = output.shape[-1] * 2
    blocks_per_col_in = n // BLOCK_SIZE
    blocks_per_col_out = physical_n // BLOCK_SIZE
    num_output_blocks = num_row * blocks_per_col_out

    output_i32 = output.view(torch.int32)
    output_scale_fp8 = output_scale.view(torch.float8_e4m3fn)

    grid = lambda meta: (
        triton.cdiv(num_output_blocks, meta["BLOCKS_PER_PROGRAM"]),
    )
    _scalesweep_mse_nvfp4_quant_kernel[grid](
        input,
        output_scale_fp8,
        output_i32,
        input_scale,
        num_output_blocks,
        NUM_ROW=num_row,
        BLOCKS_PER_COL_IN=blocks_per_col_in,
        BLOCKS_PER_COL_OUT=blocks_per_col_out,
        LOWER_BOUND=LOWER_BOUND,
        NUM_CANDIDATES=UPPER_BOUND - LOWER_BOUND + 1,
        MAX_SCALE_RAW=REF_MAX_SCALE_RAW,
        IS_SWIZZLE_SCALE=is_sf_swizzled_layout,
        BLOCKS_PER_COL_OUT_PAD=round_up(blocks_per_col_out, 4),
    )


def scalesweep_mse_nvfp4_quant_impl(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    padded_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = input.shape
    output, output_scale = create_fp4_output_tensors(
        m,
        n,
        input.device,
        is_sf_swizzled_layout,
        padded_n=padded_n,
    )
    scalesweep_mse_nvfp4_quant_out(
        input,
        input_scale,
        is_sf_swizzled_layout,
        output=output,
        output_scale=output_scale,
    )
    return output, output_scale.view(torch.float8_e4m3fn)


def _scalesweep_mse_nvfp4_quant_fake(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    padded_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = input.shape[-1]
    m = input.numel() // n
    output, output_scale = create_fp4_output_tensors(
        m,
        n,
        input.device,
        is_sf_swizzled_layout,
        padded_n=padded_n,
    )
    return output, output_scale.view(torch.float8_e4m3fn)


def _scalesweep_mse_nvfp4_quant_out_fake(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    *,
    output: torch.Tensor,
    output_scale: torch.Tensor,
) -> None:
    return None


direct_register_custom_op(
    "scalesweep_mse_nvfp4_quant",
    scalesweep_mse_nvfp4_quant_impl,
    fake_impl=_scalesweep_mse_nvfp4_quant_fake,
)

direct_register_custom_op(
    "scalesweep_mse_nvfp4_quant.out",
    scalesweep_mse_nvfp4_quant_out,
    fake_impl=_scalesweep_mse_nvfp4_quant_out_fake,
)


def scalesweep_mse_nvfp4_quant(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    padded_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    _, n = input.shape

    assert n % BLOCK_SIZE == 0, (
        f"last dim has to be multiple of {BLOCK_SIZE}, but got {n}."
    )
    assert input.dtype in (torch.float16, torch.bfloat16), (
        f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."
    )
    assert input_scale.dtype == torch.float32, (
        f"input_scale.dtype needs to be fp32 but got {input_scale.dtype}."
    )
    if padded_n is not None:
        assert padded_n >= n, f"padded_n must be >= n, got padded_n={padded_n}, n={n}."
        assert padded_n % BLOCK_SIZE == 0, (
            f"padded_n has to be a multiple of {BLOCK_SIZE}, but got {padded_n}."
        )

    return torch.ops.vllm.scalesweep_mse_nvfp4_quant(
        input.contiguous(),
        input_scale,
        is_sf_swizzled_layout,
        padded_n,
    )
