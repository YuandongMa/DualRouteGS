"""Order-preserving affine scans: h_i = a_i h_{i-1} + b_i, h_0 = 0."""
import torch


def affine_scan(a, b):
    """Inclusive associative scan; differentiable, O(N log N) reference work."""
    if a.shape != b.shape:
        raise ValueError("a and b must have the same [B,N,D] shape")
    shift = 1
    while shift < a.shape[1]:
        b = torch.cat([b[:, :shift], b[:, shift:] + a[:, shift:] * b[:, :-shift]], 1)
        a = torch.cat([a[:, :shift], a[:, shift:] * a[:, :-shift]], 1)
        shift *= 2
    return b


def compact_subspace_scan(decay, write, hard_mask, subspace_dim):
    """Inference: skip inactive writes, preserving scan order within each block.

    Tokens must NOT be sorted by rank: that would change the recurrence.
    Copy states to skipped positions by indexing the last active predecessor.
    """
    batch, tokens, width = decay.shape
    outputs = []
    for bi in range(batch):
        blocks = []
        for j in range(width // subspace_dim):
            active = hard_mask[bi, :, j].bool()
            sl = slice(j*subspace_dim, (j+1)*subspace_dim)
            ids = active.nonzero(as_tuple=True)[0]
            if ids.numel() == 0:
                blocks.append(write.new_zeros(tokens, subspace_dim))
                continue
            history = affine_scan(decay[bi:bi+1, ids, sl], write[bi:bi+1, ids, sl])[0]
            history = torch.cat([write.new_zeros(1, subspace_dim), history], 0)
            blocks.append(history[active.long().cumsum(0)])
        outputs.append(torch.cat(blocks, -1))
    return torch.stack(outputs)
