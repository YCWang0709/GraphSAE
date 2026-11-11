"""Surviving-neighbor accumulation for numerically stable mean-GraphSAGE replay."""

from contextlib import contextmanager
import types
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


@contextmanager
def stable_sage_deletion(model):
    """Accumulate surviving features in float64, then use the original modules.

    No cached-sum subtraction is used. This context is intended for input-deletion
    evaluation on large mean-GraphSAGE fields; topology is supplied by the caller.
    """
    originals = {}

    def forward(self, x, edge_index, size=None):
        if not isinstance(x, torch.Tensor) or self.project or self.aggr != "mean":
            raise ValueError(
                "Stable replay supports homogeneous unprojected mean-GraphSAGE"
            )
        source, target = edge_index
        total = torch.zeros(x.size(0), x.size(1), device=x.device, dtype=torch.float64)
        total.index_add_(0, target, x[source].double())
        degree = torch.bincount(target, minlength=x.size(0)).clamp_min(1).double()
        mean = (total / degree[:, None]).to(x.dtype)
        output = self.lin_l(mean)
        if self.root_weight:
            output = output + self.lin_r(x)
        return F.normalize(output, p=2.0, dim=-1) if self.normalize else output

    try:
        for module in model.modules():
            if isinstance(module, SAGEConv):
                originals[module] = module.forward
                module.forward = types.MethodType(forward, module)
        yield
    finally:
        for module, original in originals.items():
            module.forward = original
