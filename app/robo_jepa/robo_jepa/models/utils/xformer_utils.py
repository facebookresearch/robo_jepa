# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

try:
    from xformers.sparse.utils import _nonzero_mask_to_sparse_csr_indices
except Exception:
    pass


def build_local_attention_mask(
    T,
    H,
    W,
    V=1,
    local_window_time=-1,
    local_window_h=-1,
    local_window_w=-1,
    action_tokens=None,
    state_tokens=None,
    block_size_for_padding=None,
):
    """
    Args:
        local_window_time: Local attention temporal window size in the past
        local_window_h: Local attention kernel size in height
        local_window_w: Local attention kernel size in width
    """
    N_T = H * W * V
    if action_tokens is not None:
        N_T += action_tokens
    if state_tokens is not None:
        N_T += state_tokens
    N = T * N_T
    mask = torch.zeros(N, N).bool()
    mask_block = torch.zeros(N_T, N_T).bool()

    assert local_window_h % 2 == 1, "local_window_h must be odd"
    assert local_window_w % 2 == 1, "local_window_w must be odd"
    assert local_window_h < 0 or local_window_h > 2, "local_window_h must be <0 or >2"
    assert local_window_w < 0 or local_window_w > 2, "local_window_w must be <0 or >2"

    if local_window_time < 0:
        local_window_time = T

    if local_window_h < 0:
        local_window_h = H
    else:
        local_window_h = local_window_h // 2

    if local_window_w < 0:
        local_window_w = W
    else:
        local_window_w = local_window_w // 2

    if local_window_w == W and local_window_h == H:
        mask_block = torch.ones(N_T, N_T).bool()
    else:
        for i in range(N_T):
            h1, w1 = i // W, i % W
            for h2 in range(max(0, h1 - local_window_h), min(H, h1 + local_window_h + 1)):
                for w2 in range(max(0, w1 - local_window_w), min(W, w1 + local_window_w + 1)):
                    j = h2 * W + w2
                    mask_block[i][j] = True

    for t1 in range(T):
        for t2 in range(max(0, t1 - local_window_time + 1), t1 + 1):
            mask[t1 * N_T : (t1 + 1) * N_T, t2 * N_T : (t2 + 1) * N_T] = mask_block

    if block_size_for_padding is not None:
        pad_size = block_size_for_padding - (N % block_size_for_padding)
        if N % block_size_for_padding != 0:
            full_size = N + pad_size
            full_mask = torch.zeros(full_size, full_size).bool()
            full_mask[:N, :N] = mask
            full_mask = full_mask.contiguous()
            mask = full_mask
    return mask


# The version of this in xformers.sparse.utils casts values to float, I don't know why
def _dense_to_sparse(matrix, device):
    """Converts dense 2d matrix to a csr sparse matrix."""

    assert len(matrix.shape) == 2
    value_dtype = torch.int32

    # Extract the nonzero values.
    mask = matrix != 0
    values = matrix[mask].to(dtype=value_dtype, device=device)

    row_indices, row_offsets, column_indices = _nonzero_mask_to_sparse_csr_indices(mask, device)
    return values, row_indices, row_offsets, column_indices


def make_sparse_mask(masks, device=None):
    """Given a dense matrix, returns the CSR format TODO fill this out"""
    row_offsets = []
    column_indices = []
    column_values = []
    total = 0
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    bs = masks.shape[0]
    device = masks.device if device is None else device
    if masks.stride(0) == 0:
        masks = masks[:1]
    for mask in masks:
        cv, _, ro, ci = _dense_to_sparse(mask, mask.device)
        row_offsets.append(ro + total)
        column_indices.append(ci)
        column_values.append(cv)
        total += ro[-1]

    row_offsets = torch.stack(row_offsets, 0).to(device).expand(bs, -1)
    column_indices = torch.cat(column_indices, 0).to(device)
    column_values = torch.cat(column_values, 0).to(device)
    return row_offsets, column_indices, column_values


def _dedup_blocks(blocks: torch.Tensor, block_size: int):
    """Deduplicate a stack of (n, bs, bs) bool blocks into unique patterns.

    Returns ``(mask_blocks, inverse)`` where ``mask_blocks`` is ``(n_pat, bs,
    bs)`` (one representative per unique pattern) and ``inverse`` is ``(n,)``
    mapping every input block to its pattern id. The all-False (empty) block is
    forced to index 0 so downstream code can treat ``id != 0`` as "non-empty".

    Uses exact int64 bit-packing (63 bits/word) so the dedup runs over far
    fewer columns than the raw ``bs*bs`` bool rows.
    """
    bs = block_size
    dev = blocks.device
    n = int(blocks.shape[0])
    flat = blocks.reshape(n, bs * bs)
    nbits = bs * bs
    words = (nbits + 62) // 63
    pad = words * 63 - nbits
    packed_in = flat.to(torch.int64)
    if pad:
        packed_in = torch.cat([packed_in, torch.zeros(n, pad, dtype=torch.int64, device=dev)], dim=1)
    weights = (1 << torch.arange(63, dtype=torch.int64, device=dev))
    packed = (packed_in.reshape(n, words, 63) * weights).sum(dim=2)  # (n, words)
    _uniq, inverse = torch.unique(packed, dim=0, return_inverse=True)
    inverse = inverse.reshape(n)
    n_pat = int(_uniq.shape[0])

    # Pick one representative (bs, bs) block per unique pattern id.
    rep = torch.zeros(n_pat, dtype=torch.long, device=dev)
    rep[inverse.to(torch.long)] = torch.arange(n, device=dev)
    mask_blocks = flat[rep].reshape(n_pat, bs, bs)

    # Force the all-False (empty) block to live at index 0.
    is_zero = ~mask_blocks.reshape(n_pat, -1).any(dim=1)
    if bool(is_zero.any()):
        zidx = int(torch.nonzero(is_zero, as_tuple=False)[0].item())
        if zidx != 0:
            mask_blocks = mask_blocks.clone()
            mask_blocks[[0, zidx]] = mask_blocks[[zidx, 0]]
            new_inv = inverse.clone()
            new_inv[inverse == 0] = zidx
            new_inv[inverse == zidx] = 0
            inverse = new_inv
    else:
        zeros = torch.zeros(1, bs, bs, dtype=mask_blocks.dtype, device=mask_blocks.device)
        mask_blocks = torch.cat([zeros, mask_blocks], dim=0)
        inverse = inverse + 1

    return mask_blocks, inverse


def mask_to_blockwise(mask: torch.Tensor, block_size: int):
    """Converts an arbitrarily-sized, bool-valued Tensor to block-size masks.

    Input:
      - mask: a 2d bool-valued Tensor
    Returns:
      - masks: a Tensor of shape (N, block_size, block_size) of each unique mask pattern
      - mask_indices: a 2d int-valued Tensor, where the (i, j)th element points to the same pattern
            found at `mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size]

    Vectorized: dedup block patterns via ``torch.unique(dim=0)`` instead of a
    per-block Python hash loop. This is O(M*N) blocks done in a few tensor ops
    rather than ~M*N Python iterations each hashing block_size**2 bits — the
    Python path took ~15 min to pre-warm a 480x640 (30x40 grid) predictor; the
    vectorized path is seconds. The empty (all-False) block is forced to index
    0 to preserve the downstream convention that ``mask_indices != 0`` marks a
    non-empty block (see ``make_sparse_mask``).
    """

    assert mask.ndim == 2
    # someday this will work with funny sizes, but today is not that day
    assert (mask.shape[0] % block_size == 0) and (mask.shape[1] % block_size == 0)

    M, N = mask.shape[0] // block_size, mask.shape[1] // block_size
    bs = block_size
    # (M*bs, N*bs) -> (M, bs, N, bs) -> (M, N, bs, bs) -> (M*N, bs, bs)
    blocks = mask.reshape(M, bs, N, bs).permute(0, 2, 1, 3).reshape(M * N, bs, bs)
    mask_blocks, inverse = _dedup_blocks(blocks, bs)
    inverse = inverse.reshape(M, N).to(torch.int32)
    return mask_blocks, inverse


def build_blockwise_local_attention_mask(
    T,
    H,
    W,
    V=1,
    local_window_time=-1,
    local_window_h=-1,
    local_window_w=-1,
    action_tokens=None,
    state_tokens=None,
    block_size=64,
    device="cpu",
):
    """Build the blockwise local-attention mask WITHOUT materializing N x N.

    Semantically identical to ``mask_to_blockwise(build_local_attention_mask(
    ..., block_size_for_padding=block_size), block_size)`` but never allocates
    the dense ``(N, N)`` bool tensor (``N = T * (H*W*V + action + state)``). For
    a 720p / 3-view / 16-frame predictor that dense mask is ~30 GB and is built
    on every rank at construction time, OOMing host RAM.

    Only the per-frame-pair spatial block ``mask_block`` (``N_T x N_T``, already
    ~T**2 smaller than N x N, and skipped entirely when the spatial window is
    full) is materialized. The full mask is assembled one block-row strip
    (``block_size x N_pad``, ~a few MB) at a time; each strip is reduced to its
    block patterns and everything is globally deduplicated at the end.

    Returns ``(mask_blocks, mask_indices)`` matching ``mask_to_blockwise``:
      - ``mask_blocks``: ``(n_pat, block_size, block_size)`` unique patterns,
        all-False pattern at index 0.
      - ``mask_indices``: ``(M, M)`` int32 block-grid, ``!= 0`` marks non-empty.
    """
    bs = int(block_size)
    N_T = H * W * V
    if action_tokens is not None:
        N_T += action_tokens
    if state_tokens is not None:
        N_T += state_tokens
    N = T * N_T

    # Mirror build_local_attention_mask's window handling/asserts exactly.
    assert local_window_h % 2 == 1, "local_window_h must be odd"
    assert local_window_w % 2 == 1, "local_window_w must be odd"
    assert local_window_h < 0 or local_window_h > 2, "local_window_h must be <0 or >2"
    assert local_window_w < 0 or local_window_w > 2, "local_window_w must be <0 or >2"

    lwt = T if local_window_time < 0 else local_window_time
    lwh = H if local_window_h < 0 else local_window_h // 2
    lww = W if local_window_w < 0 else local_window_w // 2

    dev = torch.device(device)
    full_spatial = (lwh == H and lww == W)
    if full_spatial:
        mask_block = None  # implicit all-True (N_T x N_T)
    else:
        mask_block = torch.zeros(N_T, N_T, dtype=torch.bool, device=dev)
        for i in range(N_T):
            h1, w1 = i // W, i % W
            for h2 in range(max(0, h1 - lwh), min(H, h1 + lwh + 1)):
                for w2 in range(max(0, w1 - lww), min(W, w1 + lww + 1)):
                    mask_block[i, h2 * W + w2] = True

    # Padded square length (mirror block_size_for_padding).
    Np = N if N % bs == 0 else N + (bs - N % bs)
    M = Np // bs

    tt = torch.arange(T, device=dev)
    # Assemble the mask one chunk of block-rows at a time, bounding peak memory
    # to ~a few hundred MB (vs the ~30 GB dense N x N). Built directly on the
    # target device, so no huge CPU tensor is ever allocated (the legacy dense
    # path allocated the (N, N) mask on the CPU regardless of `device`).
    # Chunking many block-rows per iteration cuts Python/torch.unique overhead.
    _MEM_BUDGET = 256 * 1024 * 1024  # bytes for the transient bool strip
    rows_per_chunk = max(bs, (_MEM_BUDGET // max(1, Np)))
    rows_per_chunk = max(bs, (rows_per_chunk // bs) * bs)  # multiple of bs

    chunk_blocks = []   # list of (n_c, bs, bs) unique-per-chunk patterns
    chunk_inv = []      # list of (k_c, M) local pattern ids per block-row
    for start in range(0, Np, rows_per_chunk):
        nrows = min(rows_per_chunk, Np - start)            # multiple of bs
        r = start + torch.arange(nrows, device=dev)        # (nrows,) global rows
        valid = r < N
        r_c = torch.where(valid, r, torch.zeros_like(r))
        t1 = r_c // N_T                                    # (nrows,)
        s1 = r_c % N_T                                     # (nrows,)
        # temporal-allowed: t2 in (t1 - lwt, t1]
        allowed = (tt[None, :] <= t1[:, None]) & (tt[None, :] > (t1[:, None] - lwt)) & valid[:, None]  # (nrows,T)
        if full_spatial:
            sp = valid[:, None].expand(nrows, N_T)         # (nrows, N_T)
        else:
            sp = mask_block[s1] & valid[:, None]           # (nrows, N_T)
        # (nrows, T, N_T) -> (nrows, N) -> pad to (nrows, Np)
        strip = (allowed[:, :, None] & sp[:, None, :]).reshape(nrows, N)
        if Np > N:
            strip = torch.cat([strip, torch.zeros(nrows, Np - N, dtype=torch.bool, device=dev)], dim=1)
        # (k*bs, Np) -> (k, M, bs, bs) block patterns for these block-rows
        k = nrows // bs
        blk = strip.reshape(k, bs, M, bs).permute(0, 2, 1, 3).reshape(k * M, bs, bs)
        mb_c, inv_c = _dedup_blocks(blk, bs)               # inv_c: (k*M,)
        chunk_blocks.append(mb_c)
        chunk_inv.append(inv_c.reshape(k, M))

    # Global dedup across all chunks.
    counts = [int(b.shape[0]) for b in chunk_blocks]
    offsets = torch.zeros(len(counts), dtype=torch.long, device=dev)
    if len(counts) > 1:
        offsets[1:] = torch.tensor(counts[:-1], dtype=torch.long, device=dev).cumsum(0)
    all_blocks = torch.cat(chunk_blocks, 0)                # (sum n_c, bs, bs)
    flat_local = torch.cat([chunk_inv[c].to(torch.long) + int(offsets[c]) for c in range(len(counts))], 0)  # (M, M)
    mask_blocks, inverse = _dedup_blocks(all_blocks, bs)   # inverse: (sum n_c,)
    mask_indices = inverse[flat_local].to(torch.int32)     # (M, M)
    return mask_blocks, mask_indices
