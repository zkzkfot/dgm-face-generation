# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Custom replacement for `torch.nn.functional.grid_sample` that
supports arbitrarily high order gradients between the input and output.
Only works on 2D images and assumes
`mode='bilinear'`, `padding_mode='zeros'`, `align_corners=False`.

PATCHED for PyTorch 2.x: the custom op is force-enabled for all versions and the
internal `aten::grid_sampler_2d_backward` call uses the modern signature that
requires an `output_mask` (bool[2]) argument. This restores ADA's R1 double
backward through geometric augmentation on PyTorch 2.6 where the native
`grid_sampler_2d_backward` has no registered derivative.
"""

import warnings
import torch

# pylint: disable=redefined-builtin
# pylint: disable=arguments-differ
# pylint: disable=protected-access

#----------------------------------------------------------------------------

enabled = True  # PATCHED: enable the custom op.

#----------------------------------------------------------------------------

def grid_sample(input, grid):
    if _should_use_custom_op():
        return _GridSample2dForward.apply(input, grid)
    return torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)

#----------------------------------------------------------------------------

def _should_use_custom_op():
    return enabled  # PATCHED: use the custom op on every torch version.

#----------------------------------------------------------------------------

class _GridSample2dForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid):
        assert input.ndim == 4
        assert grid.ndim == 4
        output = torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        ctx.save_for_backward(input, grid)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grad_input, grad_grid = _GridSample2dBackward.apply(grad_output, input, grid)
        return grad_input, grad_grid

#----------------------------------------------------------------------------

class _GridSample2dBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, input, grid):
        # PATCHED: modern aten signature takes output_mask (bool[2]).
        grad_input, grad_grid = torch.ops.aten.grid_sampler_2d_backward(
            grad_output, input, grid, 0, 0, False, [True, True])
        ctx.save_for_backward(grid)
        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad2_grad_input, grad2_grad_grid):
        _ = grad2_grad_grid  # unused
        grid, = ctx.saved_tensors
        grad2_grad_output = None
        grad2_input = None
        grad2_grid = None

        if ctx.needs_input_grad[0]:
            grad2_grad_output = _GridSample2dForward.apply(grad2_grad_input, grid)

        assert not ctx.needs_input_grad[2]

        return grad2_grad_output, grad2_input, grad2_grid

#----------------------------------------------------------------------------
