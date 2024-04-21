import torch
from speculative import create_models
from utils import sample
import argparse

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

def _make_tree_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device
):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default="c4", help='dataset')
parser.add_argument('--M', type=int, default=512, help='max length')
parser.add_argument('--temperature', type=float, default=0.6, help='temperature')
parser.add_argument('--top_k', type=int, default=20, help='top_k')
parser.add_argument('--top_p', type=float, default=0.9, help='top_p')
parser.add_argument('--tree_depth', type=int, default=5, help='top_p')
args = parser.parse_args()
print(args)

MAX_LEN= args.M
DEC_LEN = 1
DTYPE = torch.float16
DEVICE = "cuda:0"
top_k=args.top_k
top_p=args.top_p
temperature=args.temperature

target, draft_1, draft_2, tokenizer = create_models(MAX_LEN)
attention_mask = _make_causal_mask((MAX_LEN, MAX_LEN), dtype=DTYPE, device=DEVICE)
attention_mask = attention_mask[None, None, :, :]

# Replace this for dataset loading
prompt= "Pittsburgh is a city located in Pennsylvania, "

input_ids=tokenizer.encode(prompt,return_tensors="pt").to(DEVICE)
PREFIX_LEN= input_ids.size(1)
position_ids = torch.arange(PREFIX_LEN, device=DEVICE).unsqueeze(0)
prefix_storage_ids = torch.arange(PREFIX_LEN, device=DEVICE)
# Prefill target model and draft model, get the bonus token from target model
logits=draft_2.inference(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., :PREFIX_LEN,:], storage_ids=prefix_storage_ids)
draft_1.inference(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., :PREFIX_LEN,:], storage_ids=prefix_storage_ids)

next_token= sample(logits[:,-1],top_k=top_k,top_p=top_p,temperature=temperature)
next_token=next_token.unsqueeze(0)
seq_offset=PREFIX_LEN
output=torch.cat([input_ids.clone(),next_token.clone()],dim=-1)

while output.size(1)<MAX_LEN:
    input_ids = next_token
    storage_ids = torch.arange(DEC_LEN, device=DEVICE) + seq_offset
    position_ids = storage_ids.clone().unsqueeze(0)
    logits=target.inference(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., seq_offset: seq_offset + DEC_LEN,:].clone(), storage_ids=storage_ids)
    next_token= sample(logits[:,-1],top_k=top_k,top_p=top_p,temperature=temperature)
    if next_token[0] == 2:
        break
    next_token=next_token.unsqueeze(0)
    output=torch.cat([output,next_token],dim=-1)
    seq_offset+=1

print(tokenizer.decode(output[0]))


# 测试正常输出，测试stage1-stage2 tree， stage2-target spec