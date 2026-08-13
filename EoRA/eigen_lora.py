import torch


def compute_eora_lora(delta, raw_scaling_diag_matrix, rank, dtype):
    """Compute low-rank EoRA A/B from a weight residual and activation covariance.

    Tiny or slightly negative eigenvalues from float32 covariance accumulation
    become arbitrarily large outliers in 1/sqrt(lambda) and propagate as NaN/Inf
    into the low-rank factors. Truncate the spectrum at a numerical-rank
    tolerance derived from float32 epsilon before taking sqrt/rsqrt so only
    well-conditioned modes contribute to the pseudoinverse.
    """

    L, Q = torch.linalg.eigh(raw_scaling_diag_matrix)

    if not torch.isfinite(L).all():
        raise FloatingPointError("EoRA covariance eigensolve produced non-finite eigenvalues.")

    relative_tolerance = min(1.0, max(1, L.numel()) * torch.finfo(torch.float32).eps)
    maximum = L[-1].clamp_min(0.0)
    cutoff = maximum * relative_tolerance
    retained = L > cutoff

    discarded_count = int((~retained).sum().item())
    if discarded_count:
        negative_count = int((L < 0).sum().item())
        print(
            f"EoRA: discarding {discarded_count}/{L.numel()} eigenvalues at or below {cutoff.item():.3e} "
            f"(negative={negative_count}, min={L[0].item():.3e}, max={L[-1].item():.3e})."
        )

    sqrtEigenvalues = torch.zeros_like(L)
    sqrtEigenvalues[retained] = torch.sqrt(L[retained])

    invSqrtEigenvalues = torch.zeros_like(L)
    invSqrtEigenvalues[retained] = torch.rsqrt(L[retained])

    scaling_diag_matrix = Q * sqrtEigenvalues.unsqueeze(0)
    scaling_matrix_inv = invSqrtEigenvalues.unsqueeze(1) * Q.T

    scaling_diag_matrix = scaling_diag_matrix.float()
    scaling_matrix_inv = scaling_matrix_inv.float()

    delta_scale = torch.matmul(delta.to(torch.float32), scaling_diag_matrix)

    U, S, VT = torch.linalg.svd(delta_scale, full_matrices=False)
    num_s_after_trunc = rank
    truc_s = S[:num_s_after_trunc]
    truc_u = U[:, :num_s_after_trunc]
    truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
    truc_sigma = torch.diag(truc_s)

    sqrtSigma = torch.sqrt(truc_sigma)
    B = torch.matmul(truc_u, sqrtSigma).to(dtype)
    A = torch.matmul(sqrtSigma, truc_v).to(dtype)
    return A, B
