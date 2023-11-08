# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mathy utility functions."""

import torch
import numpy as np


def safe_trig_helper(x, fn, t=100 * torch.pi):
    """Helper function used by safe_cos/safe_sin: mods x before sin()/cos()."""
    return fn(torch.where(torch.abs(x) < t, x, x % t))


def safe_cos(x):
    """torch.cos() on a TPU may NaN out for large values."""
    return safe_trig_helper(x, torch.cos)


def safe_sin(x):
    """torch.sin() on a TPU may NaN out for large values."""
    return safe_trig_helper(x, torch.sin)


def log_lerp(t, v0, v1):
    """Interpolate log-linearly from `v0` (t=0) to `v1` (t=1)."""
    if v0 <= 0 or v1 <= 0:
        raise ValueError(f'Interpolants {v0} and {v1} must be positive.')
    lv0 = np.log(v0)
    lv1 = np.log(v1)
    return np.exp(np.clip(t, 0, 1) * (lv1 - lv0) + lv0)


def learning_rate_decay(step,
                        lr_init,
                        lr_final,
                        max_steps,
                        lr_delay_steps=0,
                        lr_delay_mult=1):
    """Continuous learning rate decay function.
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    Args:
      step: int, the current optimization step.
      lr_init: float, the initial learning rate.
      lr_final: float, the final learning rate.
      max_steps: int, the number of steps during optimization.
      lr_delay_steps: int, the number of steps to delay the full learning rate.
      lr_delay_mult: float, the multiplier on the rate when delaying it.
    Returns:
      lr: the learning for current step 'step'.
    """
    if lr_delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
    else:
        delay_rate = 1.
    return delay_rate * log_lerp(step / max_steps, lr_init, lr_final) / lr_init


# def interp(*args):
#     """A gather-based (GPU-friendly) vectorized replacement for torch.interp()."""
#     args_flat = [x.reshape([-1, x.shape[-1]]) for x in args]
#     ret = functorch.vmap(torch.interp)(*args_flat).reshape(args[0].shape)
#     return ret


def sorted_interp(x, xp, fp):
    """A TPU-friendly version of interp(), where xp and fp must be sorted."""

    # Identify the location in `xp` that corresponds to each `x`.
    # The final `True` index in `mask` is the start of the matching interval.
    mask = x[..., None, :] >= xp[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        # print(torch.where(mask, x[..., None], x[..., :1, None]))
        # print(x[..., None].shape, x[..., :1, None].shape, torch.where(mask, x[..., None], x[..., :1, None]).shape)
        x0 = torch.max(torch.where(
            mask, x[..., None], x[..., :1, None]), dim=-2).values
        x1 = torch.min(torch.where(
            ~mask, x[..., None], x[..., -1:, None]), dim=-2).values
        return x0, x1

    fp0, fp1 = find_interval(fp)
    xp0, xp1 = find_interval(xp)

    offset = torch.clip(torch.nan_to_num((x - xp0) / (xp1 - xp0), 0), 0, 1)
    ret = fp0 + offset * (fp1 - fp0)
    return ret


def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points. Similar to np.interp.
    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.
    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.
    Returns:
        the interpolated values, same size as `x`.
    Details:
        Taken from issue at https://github.com/pytorch/pytorch/issues/50334
    """
    x = x.double()
    xp = xp.double()
    fp = fp.double()
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])

    indices = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
    indices = torch.clamp(indices, 0, len(m) - 1)

    return m[indices] * x + b[indices]