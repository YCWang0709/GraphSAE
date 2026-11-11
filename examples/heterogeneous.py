"""SAE training and typed source explanation with a relation-aware predictor."""

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from graphsae_repro import ActivationPack, FrozenModelAdapter, GraphSAE, SAEConfig
from graphsae_repro.adapters import heterogeneous_receptive_field


class TypedPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        relations = [("person", "uses", "item"), ("item", "used_by", "person")]
        self.conv = HeteroConv(
            {rel: SAGEConv((3, 3), 4) for rel in relations}, aggr="sum"
        )
        self.classifier = nn.Linear(4, 2)

    def hidden(self, data):
        return {
            kind: h.relu()
            for kind, h in self.conv(data.x_dict, data.edge_index_dict).items()
        }

    def forward(self, data):
        return self.classifier(self.hidden(data)["person"])


def run(*, epochs=2):
    torch.manual_seed(0)
    data = HeteroData()
    data["person"].x, data["item"].x = torch.rand(6, 3), torch.rand(4, 3)
    edge = torch.tensor([[0, 1, 2, 3, 4, 5, 1, 3], [0, 1, 2, 3, 0, 1, 2, 0]])
    data["person", "uses", "item"].edge_index = edge
    data["item", "used_by", "person"].edge_index = edge.flip(0)
    model = TypedPredictor()
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model(data)[:4], labels[:4]).backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        hidden = model.hidden(data)["person"][:4]
    pack = ActivationPack(
        hidden,
        [{"label": int(y), "split": "train"} for y in labels[:4]],
        "node",
        info={"split": "train"},
    )
    sae = GraphSAE.fit(pack, model.classifier, SAEConfig(epochs=epochs, k_active=3))
    adapter = FrozenModelAdapter(
        model,
        lambda d, t: model.hidden(d)[t[0]][t[1]],
        model,
        lambda d, t: heterogeneous_receptive_field(d, t, 1),
    )
    target = ("person", 4)
    result = sae.explain(adapter, data, target)
    original = model(data)[target[1]].detach()
    deleted = model(adapter.delete(data, result["selected_nodes"]))[target[1]].detach()
    return result, original, deleted


if __name__ == "__main__":
    result, _, _ = run()
    print(result["selected_nodes"])
