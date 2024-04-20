import torch

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask

MAX_LEN=16
DTYPE=torch.float16
DEVICE="cpu"
PREFIX_LEN=5
attention_mask = _make_causal_mask((MAX_LEN, MAX_LEN), dtype=DTYPE, device=DEVICE)
attention_mask = attention_mask[None, None, :, :]
attention_mask=attention_mask[..., :PREFIX_LEN,:]
print(attention_mask)
# 测试正常输出，测试stage1-stage2 tree， stage2-target spec