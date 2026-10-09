# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Static FP8 quantization for AOT-compiled inference.

Unlike dynamic FP8 (fp8_utils.py) which computes activation scales at runtime
(adding 3 ops per linear → 0.52x slower), static FP8 uses pre-calibrated scales
that become graph constants in the AOT-compiled program. Inductor can fold the
quantize multiply into the preceding Triton kernel → zero overhead.

Usage:
    from app.robo_jepa.inference.static_fp8 import calibrate_and_quantize

    # Calibrate with representative inputs, returns quantized model
    model = calibrate_and_quantize(model, calibration_fn, num_samples=50)
"""

import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple


FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0


class StaticFloat8Linear(nn.Module):
    """FP8 linear with pre-calibrated static activation scales.

    Weight is quantized to FP8 at construction time. The registered buffer is
    ``weight_fp8`` (row-major, shape [out_features, in_features]) which keeps
    state_dict FQNs compatible with AOTI-compiled programs. A non-registered
    attribute ``weight_fp8_t`` provides a col-major *view* of the same data
    (shape [in_features, out_features], stride (1, in_features)) required by
    ``torch._scaled_mm``.

    Activation scale is a fixed constant (no runtime amax computation).
    Both scales are registered buffers → become graph constants in AOT / torch.compile.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        output_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.output_dtype = output_dtype

        # Row-major weight buffer [out, in] — registered so state_dict FQNs
        # match AOTI programs compiled with the old ``weight_fp8`` name.
        self.register_buffer(
            "weight_fp8",
            torch.zeros(out_features, in_features, device=device, dtype=torch.float8_e4m3fn),
        )
        # Col-major VIEW used by forward / _scaled_mm (NOT a registered buffer).
        # Recreated after every _apply() call.
        self.weight_fp8_t: torch.Tensor = self.weight_fp8.t()

        self.register_buffer(
            "weight_scale", torch.ones(1, device=device, dtype=torch.float32)
        )
        self.register_buffer(
            "input_scale", torch.ones(1, device=device, dtype=torch.float32)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=output_dtype))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        input_scale: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> "StaticFloat8Linear":
        """Convert nn.Linear to StaticFloat8Linear with pre-calibrated scales."""
        fp8 = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            output_dtype=output_dtype,
        )

        # Quantize weight — store row-major [out, in] in the registered buffer
        w = linear.weight.data.float()
        w_amax = w.abs().amax()
        w_scale = (w_amax / FP8_MAX).clamp(min=1e-12)
        w_fp8 = (w / w_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)

        fp8.weight_fp8.copy_(w_fp8)               # row-major [out, in]
        fp8.weight_fp8_t = fp8.weight_fp8.t()      # col-major view [in, out]
        fp8.weight_scale.copy_(w_scale.reshape(1))
        fp8.input_scale.copy_(input_scale.reshape(1))

        if linear.bias is not None:
            fp8.bias.data.copy_(linear.bias.data.to(output_dtype))

        return fp8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)

        # Static quantization: use pre-calibrated input_scale (no runtime amax!)
        x_fp8 = (x_flat.float() / self.input_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)

        out = torch._scaled_mm(
            x_fp8,
            self.weight_fp8_t,
            scale_a=self.input_scale,
            scale_b=self.weight_scale,
            out_dtype=self.output_dtype,
            use_fast_accum=True,
        )
        # _scaled_mm may return (result, amax) tuple in some PyTorch versions
        if isinstance(out, tuple):
            out = out[0]

        out = out.reshape(*orig_shape, self.out_features)
        out = out.to(x.dtype)

        if self.bias is not None:
            out = out + self.bias

        return out

    def _apply(self, fn):
        """Override _apply to prevent .to(bf16) from converting FP8/F32 buffers.

        Preserves dtype of weight_fp8 (float8_e4m3fn) and scales (float32),
        then recreates the col-major view ``weight_fp8_t``.
        """
        saved_weight = self.weight_fp8.data.clone()
        saved_w_scale = self.weight_scale.data.clone()
        saved_i_scale = self.input_scale.data.clone()
        super()._apply(fn)
        dev = self.weight_fp8.device
        # Restore row-major FP8 weight
        self.weight_fp8.data = saved_weight.to(dev)
        # Recreate col-major view
        self.weight_fp8_t = self.weight_fp8.t()
        self.weight_scale.data = saved_w_scale.to(device=dev, dtype=torch.float32)
        self.input_scale.data = saved_i_scale.to(device=dev, dtype=torch.float32)
        return self

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, static_fp8=True"
        )


class _ActivationObserver:
    """Collects running max of activation magnitudes for calibration."""

    def __init__(self):
        self.amax: Optional[torch.Tensor] = None
        self.count = 0

    def observe(self, x: torch.Tensor):
        batch_amax = x.float().abs().amax()
        if self.amax is None:
            self.amax = batch_amax
        else:
            self.amax = torch.max(self.amax, batch_amax)
        self.count += 1

    @property
    def scale(self) -> torch.Tensor:
        if self.amax is None or self.amax == 0:
            return torch.tensor(1.0, dtype=torch.float32)
        return (self.amax / FP8_MAX).clamp(min=1e-12)


def calibrate_activation_scales(
    model: nn.Module,
    calibration_fn: Callable[[nn.Module], None],
) -> Dict[str, torch.Tensor]:
    """Run calibration to collect per-layer activation scales.

    Args:
        model: The model to calibrate (in eval mode, on GPU)
        calibration_fn: A function that runs representative forward passes
            through the model. E.g., run 50 steps with random inputs.

    Returns:
        Dict mapping linear layer name to its input activation scale.
    """
    observers: Dict[str, _ActivationObserver] = {}
    hooks: List[torch.utils.hooks.RemovableHook] = []

    # Install forward hooks on all nn.Linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            obs = _ActivationObserver()
            observers[name] = obs

            def make_hook(observer):
                def hook_fn(mod, inputs):
                    if len(inputs) > 0:
                        observer.observe(inputs[0])
                return hook_fn

            hooks.append(module.register_forward_pre_hook(make_hook(obs)))

    # Run calibration
    with torch.no_grad():
        calibration_fn(model)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Extract scales
    scales = {name: obs.scale for name, obs in observers.items()}
    print(f"Calibrated {len(scales)} layers ({sum(o.count for o in observers.values())} observations)")
    return scales


def apply_static_fp8(
    model: nn.Module,
    scales: Dict[str, torch.Tensor],
    output_dtype: torch.dtype = torch.bfloat16,
    skip_layers: Optional[List[str]] = None,
) -> Tuple[nn.Module, int]:
    """Replace nn.Linear layers with StaticFloat8Linear using calibrated scales.

    Args:
        model: The model to quantize
        scales: Per-layer activation scales from calibrate_activation_scales
        output_dtype: Output dtype for FP8 matmul
        skip_layers: Layer name patterns to skip (keep in bf16)

    Returns:
        (model, count) — the quantized model and number of layers converted
    """
    skip_layers = skip_layers or []
    count = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        # Check skip list
        if any(skip in name for skip in skip_layers):
            continue

        if name not in scales:
            continue

        # Navigate to parent module to replace the child
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent = model.get_submodule(parts[0])
            child_name = parts[1]
        else:
            parent = model
            child_name = parts[0]

        fp8_linear = StaticFloat8Linear.from_linear(
            module, input_scale=scales[name], output_dtype=output_dtype
        )
        setattr(parent, child_name, fp8_linear)
        count += 1

    return model, count


def calibrate_and_quantize(
    model: nn.Module,
    calibration_fn: Callable[[nn.Module], None],
    output_dtype: torch.dtype = torch.bfloat16,
    skip_layers: Optional[List[str]] = None,
) -> Tuple[nn.Module, int]:
    """End-to-end: calibrate activation scales, then apply static FP8.

    Args:
        model: The model to quantize (eval mode, on GPU)
        calibration_fn: Runs representative forward passes for calibration
        output_dtype: Output dtype for FP8 matmul
        skip_layers: Layer name patterns to skip

    Returns:
        (model, count) — quantized model and number of FP8 layers
    """
    scales = calibrate_activation_scales(model, calibration_fn)
    return apply_static_fp8(model, scales, output_dtype, skip_layers)
