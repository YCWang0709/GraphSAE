"""Threshold estimators for the JumpReLU encoder and its L0 objective."""

import torch


class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return x * (x > threshold)

    @staticmethod
    def backward(ctx, gradient):
        x, threshold = ctx.saved_tensors
        rectangle = ((x - threshold).abs() < ctx.bandwidth / 2).to(x.dtype)
        threshold_gradient = -gradient * threshold * rectangle / ctx.bandwidth
        return (
            gradient * (x > threshold),
            threshold_gradient.sum_to_size(threshold.shape),
            None,
        )


class Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, gradient):
        x, threshold = ctx.saved_tensors
        rectangle = ((x - threshold).abs() < ctx.bandwidth / 2).to(x.dtype)
        return (
            torch.zeros_like(x),
            (-gradient * rectangle / ctx.bandwidth).sum_to_size(threshold.shape),
            None,
        )
