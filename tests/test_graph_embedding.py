import torch
from hypsimcse.models.graph_embedding import GraphEmbedding
from hypsimcse.geometry import lorentz as L


def test_graph_embedding_hyp_on_manifold():
    m = GraphEmbedding(num_nodes=50, dim=5, geometry="HYP", c=1.0)
    z = m(torch.tensor([0, 1, 2]))
    assert z.shape == (3, 6)
    assert torch.allclose(L.lorentz_inner(z, z), torch.full((3,), -1.0), atol=1e-3)


def test_graph_embedding_grad_flows():
    m = GraphEmbedding(num_nodes=50, dim=5, geometry="HYP")
    z = m(torch.tensor([0, 1]))
    z.sum().backward()
    assert m.table.weight.grad is not None
