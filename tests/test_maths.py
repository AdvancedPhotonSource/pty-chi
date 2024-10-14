import ptychointerim.maths as pmath

import torch


def test_orthgonalize_gs():
    torch.manual_seed(123)
    x = torch.rand(2, 3, 4, 5)
    x = pmath.orthogonalize_gs(x, dim=(-2, -1), group_dim=1)
    prod = torch.sum(x[0, 0] * x[0, 1])
    assert prod < 1e-4


def test_orthgonalize_svd():
    torch.manual_seed(123)
    x = torch.rand(2, 4, 5) + 1j * torch.rand(2, 4, 5)
    x = pmath.orthogonalize_svd(x, dim=(-2, -1), group_dim=0)
    prod = torch.sum(x[0] * x[1].conj())
    assert prod.abs() < 1e-4

    x = torch.rand(2, 3, 4, 5) + 1j * torch.rand(2, 3, 4, 5)
    x = pmath.orthogonalize_svd(x, dim=(-2, -1), group_dim=1)
    prod = torch.sum(x[0, 0] * x[0, 1].conj())
    assert prod.abs() < 1e-4

    x = torch.rand(2, 3, 4, 5) + 1j * torch.rand(2, 3, 4, 5)
    x = pmath.orthogonalize_svd(x, dim=(-2, -1), group_dim=0)
    prod = torch.sum(x[0, 0] * x[1, 0].conj())
    assert prod.abs() < 1e-4


if __name__ == '__main__':
    test_orthgonalize_gs()
    test_orthgonalize_svd()