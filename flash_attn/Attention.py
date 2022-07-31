import torch

from flash_attn.flash_attn_interface import flash_attn_unpadded_func


def flash_attn(q, k, v):
    # import pdb; pdb.set_trace()
    batch_dims = q.shape[:-3]
    no_heads, n, c = q.shape[-3:]
    dtype = q.dtype

    # [*, B, N, H, C]
    q = q.transpose(-2, -3)
    k = k.transpose(-2, -3)
    v = v.transpose(-2, -3)

    # [B_flat, N, H, C]
    q = q.reshape(-1, *q.shape[-3:])
    k = k.reshape(-1, *k.shape[-3:])
    v = v.reshape(-1, *v.shape[-3:])

    # Flattened batch size
    batch_size = q.shape[0]
    
    # [B_flat * N, H, C]
    q = q.reshape(-1, *q.shape[-2:])
    k = k.reshape(-1, *k.shape[-2:])
    v = v.reshape(-1, *v.shape[-2:])
    
    q_max_s = n
    q_cu_seqlens = torch.arange(
        0, (batch_size + 1) * n, step=n, dtype=torch.int32, device=q.device
    )

    k_max_s = n
    k_cu_seqlens = torch.arange(
        0, (batch_size + 1) * n, step=n, dtype=torch.int32, device=k.device
    )

    out = flash_attn_unpadded_func(
        q,
        k,
        v,
        q_cu_seqlens,
        k_cu_seqlens,
        q_max_s,
        k_max_s,
        dropout_p = 0.,
        softmax_scale = 1., # q has been scaled already
    )
    # [*, B, N, H, C]
    out = out.reshape(*batch_dims, n, no_heads, c) 

    out = out.to(dtype=dtype)

    return out