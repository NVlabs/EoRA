import pytest
import torch
from eigen_lora import compute_eora_lora


def _random_symmetric_with_spectrum(eigenvalues: torch.Tensor) -> torch.Tensor:
    """Build a symmetric matrix with the requested eigenvalue spectrum."""

    dim = eigenvalues.numel()
    Q, _ = torch.linalg.qr(torch.randn(dim, dim, dtype=torch.float64))
    return Q @ torch.diag(eigenvalues.to(dtype=torch.float64)) @ Q.T


@pytest.mark.parametrize("dim", [8, 16])
@pytest.mark.parametrize("rank", [2, 4])
def test_compute_eora_lora_clamps_negative_and_tiny_eigenvalues(dim, rank):
    """A covariance with negative/tiny eigenvalues must not produce Inf/NaN adapters."""

    torch.manual_seed(0)
    device = torch.device("cpu")

    eigenvalues = torch.zeros(dim, dtype=torch.float64, device=device)
    eigenvalues[-1] = 1.0
    eigenvalues[-2] = 1e-12
    eigenvalues[0] = -1e-8
    eigenvalues[1] = -1e-6

    eigen_scaling = _random_symmetric_with_spectrum(eigenvalues)
    delta = torch.randn(8, dim, dtype=torch.float32, device=device)

    A, B = compute_eora_lora(
        delta,
        eigen_scaling,
        rank=rank,
        dtype=torch.float16,
    )

    assert A.shape == (rank, dim)
    assert B.shape == (8, rank)
    assert torch.isfinite(A).all()
    assert torch.isfinite(B).all()
    assert torch.isfinite(B @ A).all()


def test_compute_eora_lora_handles_all_negative_eigenvalues():
    """A pathological negative-definite covariance must not crash or return non-finite values."""

    torch.manual_seed(0)
    device = torch.device("cpu")
    dim = 8
    rank = 4

    eigenvalues = -torch.logspace(-8, -2, dim, device=device)
    eigen_scaling = _random_symmetric_with_spectrum(eigenvalues)
    delta = torch.randn(8, dim, dtype=torch.float32, device=device)

    A, B = compute_eora_lora(
        delta,
        eigen_scaling,
        rank=rank,
        dtype=torch.float32,
    )

    assert A.shape == (rank, dim)
    assert B.shape == (8, rank)
    assert torch.isfinite(A).all()
    assert torch.isfinite(B).all()
    assert torch.isfinite(B @ A).all()
