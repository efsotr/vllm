# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest
import torch

from vllm import _custom_ops as ops
from vllm._custom_ops import create_fp4_output_tensors
from vllm.model_executor.layers.quantization.utils.scalesweep_mse_nvfp4_utils import (
    BLOCK_SIZE,
    FP4_E2M1_MAX,
    LOWER_BOUND,
    REF_MAX_SCALE_RAW,
    UPPER_BOUND,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.math_utils import round_up

if (
    not bool(int(os.getenv("SCALESWEEP_MSE_EMULATION", "0")))
    and not current_platform.has_device_capability(100)
):
    pytest.skip(
        reason="NVFP4 requires compute capability of 10 or above.",
        allow_module_level=True,
    )

DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [(1, 16), (3, 64), (32, 128), (128, 64), (150, 80)]
PADDED_OUTPUT_SHAPES = [(1, 16), (32, 48), (90, 80), (128, 48), (150, 80)]
CUDA_DEVICES = ["cuda:0"]

REF_FP8_E4M3_MAX = 256.0

E2M1_TO_FLOAT32 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]

BOUND_EXAMPLES = [
    (0x35, (
        6, 0.405, 0.405, 0.405, 0.405, 0.405, 0.405, 0.405,
        0.405, 4.435, 4.94, 4.94, 4.94, 4.94, 4.94, 4.94,
    )),
    (0x37, (
        6.75, 0.43875, 0.43875, 0.43875, 0.43875, 0.43875, 0.43875, 0.43875,
        0.43875, 5.068125, 5.248125, 5.563125, 5.563125, 5.563125, 6.06375, 6.06375,
    )),
    (0x38, (
        7.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 5.80625, 5.80625, 5.80625, 6.2375, 6.50625,
    )),
    (0x39, (
        8.25, 0.56375, 0.56375, 0.56375, 0.56375, 0.56375, 0.56375, 0.56375,
        0.56375, 0.56375, 4.310625, 6.373125, 6.373125, 6.620625, 7.1225, 7.1225,
    )),
    (0x3A, (
        9, 0.6225, 0.6225, 0.6225, 0.6225, 0.6225, 0.6225, 0.6225,
        0.6225, 4.8075, 4.8075, 7.2525, 7.2525, 7.2525, 7.8675, 7.8675,
    )),
    (0x3B, (
        9.75, 0.6825, 0.6825, 0.6825, 0.6825, 0.6825, 0.6825, 0.6825,
        0.6825, 5.24875, 5.24875, 7.62125, 7.873125, 8.636875, 8.636875, 8.636875,
    )),
    (0x3C, (
        10.5, 0.69125, 0.69125, 0.69125, 0.69125, 0.69125, 0.69125, 0.69125,
        5.6875, 5.6875, 8.25125, 8.61875, 8.61875, 9.37125, 9.37125, 9.37125,
    )),
    (0x3C, (
        11.25, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
        2.25, 2.25, 2.25, 8.634375, 8.634375, 8.634375, 9.50625, 9.50625,
    )),
    (0x3F, (
        5.64, 3.721, 1.887, 5.641, 5.652, 0.945, 3.743, 6.749,
        5.608, 5.656, 2.797, 1.872, 3.616, 5.665, 3.76, 1.864,
    )),
    (0x40, (
        5.99175, 6.022125, 5.99625, 0.0585, 6.364125, 4.0185, 6.50025, 0.979875,
        1.50975, 2.964375, 7.486875, 6.001875, 0, 4.026375, 6.0075, 2.92275,
    )),
    (0x41, (
        1.70375, 6.765, 0.065, 8.21875, 3.40375, 4.49, 6.78125, 6.7675,
        1.1125, 0.13125, 6.79125, 4.46375, 3.4475, 4.47875, 4.65875, 8.16125,
    )),
    (0x42, (
        7.4855, 7.418125, 0.048125, 0.034375, 8.160625, 7.48275, 2.520375, 2.569875,
        8.7835, 3.76475, 7.43325, 4.81525, 3.724875, 7.469, 7.46075, 8.98975,
    )),
    (0x43, (
        2.7855, 8.295, 8.361, 9.744, 8.2695, 1.4205, 8.136, 2.64,
        1.818, 9.6825, 5.502, 8.2605, 8.298, 1.3695, 1.2195, 2.871,
    )),
    (0x43, (
        2.770625, 8.827, 8.28425, 5.5185, 8.155875, 8.185125, 1.389375, 1.3975,
        1.378, 8.250125, 1.3975, 2.734875, 2.702375, 10.4715, 5.53475, 8.289125,
    )),
    (0x44, (
        9.07725, 8.99675, 1.56975, 6.16875, 5.908, 1.54175, 11.193, 4.4835,
        3.01, 9.156, 1.407, 9.1385, 9.1, 4.59375, 0, 6.118,
    )),
    (0x45, (
        9.795, 9.5175, 9.695625, 7.565625, 11.95875, 6.55125, 0, 1.674375,
        9.70875, 9.8325, 3.1725, 0.03, 7.306875, 4.87875, 9.73125, 6.525,
    )),
]


def cast_from_fp4(x: torch.Tensor, m: int, n: int) -> torch.Tensor:
    v_2nd = x & 0xF
    v_1st = (x >> 4) & 0xF
    c = torch.stack((v_2nd, v_1st), dim=-1)
    lut = torch.tensor(E2M1_TO_FLOAT32, device=x.device, dtype=torch.float32)
    return lut[c.long()].reshape(m, n)


def cast_to_fp4(x: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(x)
    x_abs = torch.abs(x)
    out = torch.empty_like(x_abs)
    out[(x_abs >= 0.0) & (x_abs <= 0.25)] = 0.0
    out[(x_abs > 0.25) & (x_abs < 0.75)] = 0.5
    out[(x_abs >= 0.75) & (x_abs <= 1.25)] = 1.0
    out[(x_abs > 1.25) & (x_abs < 1.75)] = 1.5
    out[(x_abs >= 1.75) & (x_abs <= 2.5)] = 2.0
    out[(x_abs > 2.5) & (x_abs < 3.5)] = 3.0
    out[(x_abs >= 3.5) & (x_abs <= 5.0)] = 4.0
    out[x_abs > 5.0] = 6.0
    return out * sign


def recover_swizzled_scales(scale: torch.Tensor, m: int, n: int) -> torch.Tensor:
    return recover_swizzled_scale_raw(scale, m, n).view(torch.float8_e4m3fn).to(
        torch.float32
    )


def recover_swizzled_scale_raw(scale: torch.Tensor, m: int, n: int) -> torch.Tensor:
    scale_n = n // BLOCK_SIZE
    rounded_m = round_up(m, 128)
    rounded_n = round_up(scale_n, 4)
    tmp = torch.reshape(scale, (1, rounded_m // 128, rounded_n // 4, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    result = torch.reshape(tmp, (rounded_m, rounded_n)).view(torch.uint8)
    return result[:m, :scale_n]


def compute_global_scale_inv(x: torch.Tensor) -> torch.Tensor:
    tensor_amax = torch.abs(x).max().to(torch.float32)
    return REF_FP8_E4M3_MAX * scalar_types.float4_e2m1f.max() / tensor_amax


def assert_scalesweep_mse_matches_ref(
    out: torch.Tensor,
    out_scale: torch.Tensor,
    x: torch.Tensor,
    out_ref: torch.Tensor,
    scale_ref: torch.Tensor,
    is_sf_swizzled_layout: bool,
    padded_n: int | None = None,
) -> None:
    m, n = x.shape
    physical_n = padded_n if padded_n is not None else n

    out_ans = cast_from_fp4(out[:, : n // 2], m, n)
    if is_sf_swizzled_layout:
        scale_raw = recover_swizzled_scale_raw(out_scale, m, physical_n)
    else:
        scale_raw = out_scale.view(torch.uint8)
    scale_ans = scale_raw[:, : n // BLOCK_SIZE].view(torch.float8_e4m3fn)

    torch.testing.assert_close(out_ans, out_ref)
    torch.testing.assert_close(scale_ans.to(torch.float32), scale_ref.to(torch.float32))

    if physical_n != n:
        assert torch.count_nonzero(out[:, n // 2 :]) == 0
        pad_raw = scale_raw[:, n // BLOCK_SIZE :]
        assert torch.all(pad_raw == 1)


def ref_scalesweep_mse_nvfp4_quant(
    x: torch.Tensor,
    global_scale_inv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert global_scale_inv.dtype == torch.float32
    assert x.ndim == 2
    m, n = x.shape
    blocks = x.reshape(m, n // BLOCK_SIZE, BLOCK_SIZE).to(torch.float32)
    blocks = blocks * global_scale_inv

    abs_max = torch.abs(blocks).amax(dim=-1)
    base_scale = abs_max * (1.0 / FP4_E2M1_MAX)
    base_raw = base_scale.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int32)
    offsets = torch.arange(
        LOWER_BOUND,
        UPPER_BOUND + 1,
        device=x.device,
        dtype=torch.int32,
    )
    scale_raw = torch.clamp(
        base_raw.unsqueeze(-1) + offsets,
        1,
        REF_MAX_SCALE_RAW,
    ).to(torch.uint8)
    scales = scale_raw.view(torch.float8_e4m3fn).to(torch.float32)

    scaled = blocks.unsqueeze(2) / scales.unsqueeze(-1)
    quantized = cast_to_fp4(scaled)
    reconstructed = quantized * scales.unsqueeze(-1)
    squared_error = torch.sum((reconstructed - blocks.unsqueeze(2)) ** 2, dim=-1)
    best_index = torch.argmin(squared_error, dim=-1)

    best_scale_raw = torch.gather(
        scale_raw,
        dim=2,
        index=best_index.unsqueeze(-1),
    ).squeeze(-1)
    best_scale = best_scale_raw.view(torch.float8_e4m3fn)
    best_quantized = torch.gather(
        quantized,
        dim=2,
        index=best_index[:, :, None, None].expand(-1, -1, 1, BLOCK_SIZE),
    ).squeeze(2)

    return best_quantized.reshape(m, n), best_scale


@torch.compile
def ref_scalesweep_mse_nvfp4_quant_all_positive_fp8(
    x: torch.Tensor,
    global_scale_inv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert global_scale_inv.dtype == torch.float32
    assert x.ndim == 2
    m, n = x.shape
    blocks = x.reshape(m, n // BLOCK_SIZE, BLOCK_SIZE).to(torch.float32)
    blocks = blocks * global_scale_inv

    scale_raw = torch.arange(
        1,
        REF_MAX_SCALE_RAW + 1,
        device=x.device,
        dtype=torch.uint8,
    )
    scales = scale_raw.view(torch.float8_e4m3fn).to(torch.float32)
    scaled = blocks.unsqueeze(2) / scales[None, None, :, None]
    quantized = cast_to_fp4(scaled)
    reconstructed = quantized * scales[None, None, :, None]
    squared_error = torch.sum((reconstructed - blocks.unsqueeze(2)) ** 2, dim=-1)
    best_index = torch.argmin(squared_error, dim=-1)

    best_scale_raw = scale_raw[best_index]
    best_scale = best_scale_raw.view(torch.float8_e4m3fn)
    best_quantized = torch.gather(
        quantized,
        dim=2,
        index=best_index[:, :, None, None].expand(-1, -1, 1, BLOCK_SIZE),
    ).squeeze(2)

    return best_quantized.reshape(m, n), best_scale


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_scalesweep_mse_nvfp4_quant(
    dtype: torch.dtype,
    shape: tuple[int, int],
    is_sf_swizzled_layout: bool,
    device: str,
) -> None:
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    global_scale_inv = compute_global_scale_inv(x)

    out_ref, scale_ref = ref_scalesweep_mse_nvfp4_quant(x, global_scale_inv)
    out, out_scale = ops.scaled_fp4_quant(
        x,
        global_scale_inv,
        is_sf_swizzled_layout=is_sf_swizzled_layout,
        backend="scalesweep_mse",
    )

    assert out.shape == (shape[0], shape[1] // 2)
    assert out.dtype == torch.uint8
    assert out_scale.dtype == torch.float8_e4m3fn

    assert_scalesweep_mse_matches_ref(
        out,
        out_scale,
        x,
        out_ref,
        scale_ref,
        is_sf_swizzled_layout,
    )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_scalesweep_mse_nvfp4_quant_default_and_out(
    shape: tuple[int, int],
    is_sf_swizzled_layout: bool,
    device: str,
) -> None:
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    x = torch.randn(shape, device=device, dtype=torch.float16, generator=generator)
    global_scale_inv = compute_global_scale_inv(x)

    out_ref, scale_ref = ref_scalesweep_mse_nvfp4_quant(x, global_scale_inv)
    out, out_scale = torch.ops.vllm.scalesweep_mse_nvfp4_quant(
        x,
        global_scale_inv,
        is_sf_swizzled_layout,
        None,
    )
    expected_out, expected_scale = create_fp4_output_tensors(
        shape[0],
        shape[1],
        torch.device(device),
        is_sf_swizzled_layout,
    )

    assert out.shape == expected_out.shape
    assert out.dtype == expected_out.dtype
    assert out_scale.shape == expected_scale.view(torch.float8_e4m3fn).shape
    assert out_scale.dtype == torch.float8_e4m3fn
    assert_scalesweep_mse_matches_ref(
        out,
        out_scale,
        x,
        out_ref,
        scale_ref,
        is_sf_swizzled_layout,
    )

    out_out = torch.empty_like(out)
    out_scale_out = torch.empty_like(expected_scale)
    torch.ops.vllm.scalesweep_mse_nvfp4_quant.out(
        x,
        global_scale_inv,
        is_sf_swizzled_layout,
        output=out_out,
        output_scale=out_scale_out,
    )
    assert_scalesweep_mse_matches_ref(
        out_out,
        out_scale_out.view(torch.float8_e4m3fn),
        x,
        out_ref,
        scale_ref,
        is_sf_swizzled_layout,
    )


@pytest.mark.parametrize("shape", PADDED_OUTPUT_SHAPES)
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_scalesweep_mse_nvfp4_quant_padded_output(
    shape: tuple[int, int],
    is_sf_swizzled_layout: bool,
    device: str,
) -> None:
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    m, n = shape
    padded_n = round_up(n + BLOCK_SIZE, 32)
    x = torch.randn(shape, device=device, dtype=torch.float16, generator=generator)
    global_scale_inv = compute_global_scale_inv(x)

    out_ref, scale_ref = ref_scalesweep_mse_nvfp4_quant(x, global_scale_inv)
    out, out_scale = ops.scaled_fp4_quant(
        x,
        global_scale_inv,
        is_sf_swizzled_layout=is_sf_swizzled_layout,
        backend="scalesweep_mse",
        padded_n=padded_n,
    )
    expected_out, expected_scale = create_fp4_output_tensors(
        m,
        n,
        torch.device(device),
        is_sf_swizzled_layout,
        padded_n=padded_n,
    )

    assert out.shape == expected_out.shape
    assert out.shape == (m, padded_n // 2)
    assert out.dtype == torch.uint8
    assert out_scale.shape == expected_scale.view(torch.float8_e4m3fn).shape
    assert out_scale.dtype == torch.float8_e4m3fn
    assert_scalesweep_mse_matches_ref(
        out,
        out_scale,
        x,
        out_ref,
        scale_ref,
        is_sf_swizzled_layout,
        padded_n=padded_n,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_scalesweep_mse_nvfp4_quant_bound_examples(
    dtype: torch.dtype,
    is_sf_swizzled_layout: bool,
    device: str,
) -> None:
    x = torch.tensor(
        [example for _, example in BOUND_EXAMPLES],
        device=device,
        dtype=dtype,
    )
    global_scale_inv = torch.tensor(1.0, device=device, dtype=torch.float32)

    out_ref, scale_ref = ref_scalesweep_mse_nvfp4_quant(x, global_scale_inv)
    out, out_scale = ops.scaled_fp4_quant(
        x,
        global_scale_inv,
        is_sf_swizzled_layout=is_sf_swizzled_layout,
        backend="scalesweep_mse",
    )

    out_ans = cast_from_fp4(out, *x.shape)
    if is_sf_swizzled_layout:
        scale_ans = recover_swizzled_scales(out_scale, *x.shape)
    else:
        scale_ans = out_scale.to(torch.float32)

    expected_raw = torch.tensor(
        [target_bit for target_bit, _ in BOUND_EXAMPLES],
        device=device,
        dtype=torch.uint8,
    ).view(torch.float8_e4m3fn)
    torch.testing.assert_close(out_ans, out_ref)
    torch.testing.assert_close(scale_ans, scale_ref.to(torch.float32))
    torch.testing.assert_close(scale_ans[:, 0], expected_raw.to(torch.float32))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_scalesweep_mse_matches_all_positive_fp8_reference(
    dtype: torch.dtype,
    device: str,
) -> None:
    generator = torch.Generator(device=device)
    generator.manual_seed(123)
    x = torch.randn((4, 64), device=device, dtype=dtype, generator=generator)
    global_scale_inv = compute_global_scale_inv(x)

    out_ref, scale_ref = ref_scalesweep_mse_nvfp4_quant(x, global_scale_inv)
    out_all, scale_all = ref_scalesweep_mse_nvfp4_quant_all_positive_fp8(
        x,
        global_scale_inv,
    )

    torch.testing.assert_close(out_ref, out_all)
    torch.testing.assert_close(scale_ref.to(torch.float32), scale_all.to(torch.float32))
