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

def make_tree_mask(seqlen_offset,depth,device, dtype, ancestors, max_len):
    attention_mask= torch.full((2**(depth), max_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    attention_mask[:,:seqlen_offset]=0.0
    for j in range(2**depth):
          ancestor=ancestors[j]
          attention_mask[j,ancestor]=0.0
    attention_mask.to(dtype)
    return attention_mask[None, None, :, :]

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default="c4", help='dataset')
parser.add_argument('--M', type=int, default=512, help='max length')
parser.add_argument('--temperature', type=float, default=0.6, help='temperature')
parser.add_argument('--top_k', type=int, default=20, help='top_k')
parser.add_argument('--top_p', type=float, default=0.9, help='top_p')
parser.add_argument('--tree_depth', type=int, default=5, help='top_p')
parser.add_argument('--gamma', type=int, default=10, help='gamma')
args = parser.parse_args()
print(args)

MAX_LEN= args.M
DEC_LEN = 1
DTYPE = torch.float16
DEVICE = "cuda:0"
DEPTH=args.tree_depth
gamma=args.gamma
top_k=args.top_k
top_p=args.top_p
temperature=args.temperature

target, draft_1, draft_2, tokenizer = create_models(MAX_LEN)
attention_mask = _make_causal_mask((MAX_LEN, MAX_LEN), dtype=DTYPE, device=DEVICE)
attention_mask = attention_mask[None, None, :, :]

# Replace this for dataset loading
prompt= "Pittsburgh is a city located in Pennsylvania, "

input_ids=tokenizer.encode(prompt,return_tensors="pt").to(DEVICE)
prefix_len= input_ids.size(1)
position_ids = torch.arange(prefix_len, device=DEVICE).unsqueeze(0)
prefix_storage_ids = torch.arange(prefix_len, device=DEVICE)
# Prefill target model and draft model, get the bonus token from target model
logits=draft_2.inference(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., :prefix_len,:], storage_ids=prefix_storage_ids)
draft_1.inference(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., :prefix_len,:], storage_ids=prefix_storage_ids)

next_token= sample(logits[:,-1],top_k=top_k,top_p=top_p,temperature=temperature)
next_token=next_token.unsqueeze(0)
# output=torch.cat([input_ids.clone(),next_token.clone()],dim=-1)
output=next_token.clone()


# Draft_1 and Draft_2 perform first stage tree speculative decoding
while output.size(1)<gamma:
    seq_offset=prefix_len
    tree=next_token.clone()
    ancestors=[[seq_offset]]
    input_ids=next_token.clone()
    i=0
    verify_mask=None
    verify_position_ids=None
    while True:
          width=input_ids.size(1)
          storage_ids=torch.arange(start=seq_offset, end=seq_offset+width ,dtype=torch.long, device=DEVICE)
          position_ids=torch.full((1,width),prefix_len+i,dtype=torch.long, device=DEVICE)
          attention_mask=make_tree_mask(seq_offset,i,DEVICE,DTYPE,ancestors,MAX_LEN)
          if verify_mask==None:
              verify_mask=attention_mask.clone()
              verify_position_ids=position_ids.clone()
          else:
              verify_mask=torch.cat((verify_mask,attention_mask),dim=-2)
              verify_position_ids=torch.cat((verify_position_ids,position_ids),dim=-1)
          logits=draft_1.inference(input_ids=input_ids,storage_ids=storage_ids,position_ids=position_ids,attention_mask=attention_mask)
          if(i==DEPTH):
               break
          top2=torch.topk(logits[:,-width:],2,dim=-1).indices.view(1,-1)
          tree=torch.cat((tree,top2),dim=-1)
          input_ids=top2
          new_ancestors=[]
          seq_offset+=width
          offset=seq_offset-1
          for j in range(width):
               offset+=1
               left=ancestors[j].copy()
               left.append(offset)

               right=ancestors[j].copy()
               offset+=1
               right.append(offset)

               new_ancestors.append(left)
               new_ancestors.append(right)
          ancestors=new_ancestors
          i+=1
    # Draft 2 Verify the tree
    storage_ids=torch.arange(start=seq_offset, end=seq_offset+2**(DEPTH+1)-1, dtype=torch.long, device=DEVICE)
    logits=draft_2.inference(input_ids=tree, storage_ids=storage_ids,position_ids=verify_position_ids,attention_mask=verify_mask)
    verified_tree=sample(logits.squeeze(0),top_k=top_k,top_p=top_p,temperature=temperature)
    saved_position_ids=[0]
    target_index=0
    bonus_token=None
    while True:
        target_id=verified_tree[target_index]
        if target_index*2+2<2**(DEPTH+1)-1:
            child_draft_1=tree[0,target_index*2+1]
            child_draft_2=tree[0,target_index*2+2]
        else:
            bonus_token=target_id
            break
        if child_draft_1==target_id:
            target_index=target_index*2+1
            saved_position_ids.append(target_index)
            continue
        if child_draft_2==target_id:
            target_index=target_index*2+2
            saved_position_ids.append(target_index)
            continue
        bonus_token=target_id
        break
    output=torch.cat((output,tree[:,saved_position_ids]),dim=-1)
    saved_position_ids= [x + prefix_len for x in saved_position_ids]
    next_token=bonus_token.view(1,1)
    draft_1.llm.kv_cache.gather_kv_incremental(saved_position_ids,prefix_len)
    draft_2.llm.kv_cache.gather_kv_incremental(saved_position_ids,prefix_len)
    prefix_len+=len(saved_position_ids)
         
print(tokenizer.decode(output[0]))

# 1 2 4 8 16 if one of 16 is accepted, we need to forward it with the bonus token again to get the correct predict