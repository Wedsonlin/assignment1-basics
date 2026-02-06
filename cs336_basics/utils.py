import torch
from jaxtyping import Float, Int
from math import cos, pi
from collections.abc import Iterable
import numpy.typing as npt
import numpy as np
import random
import os
import typing

def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    max_element = x.amax(dim=i,keepdim=True) # avoid overflow
    exp_y = torch.exp(x-max_element)
    return exp_y / exp_y.sum(dim=i,keepdim=True)

def cross_entropy(logits: Float[torch.Tensor, " batch_size vocab_size"], targets: Int[torch.Tensor, " batch_size"]
) -> Float[torch.Tensor, ""]:
    logits = logits - logits.amax(dim=-1, keepdim=True) # prevent overflow
    target_logits = logits.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1) # target_logits:(batch,)
    output = torch.log(logits.exp().sum(dim=-1)) - target_logits
    return output.mean()

def learning_rate_schedule(t: int, lr_max: float, lr_min: float, T_w: int, T_c: int) -> float:
    if t < T_w:
        lr = t/T_w*lr_max
    elif t >= T_w and t <= T_c:
        lr = lr_min + 0.5 * (1 + cos((t-T_w)/(T_c-T_w)*pi)) * (lr_max - lr_min)
    else:
        lr = lr_min
    
    return lr

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float=1e-6) -> None:
    '''
        First, calculate l2-norm of each gradient;
        Second, stack the norms of each gradient as a tensor;
        Third, calcualte l2-norm of the stacked tensor;

        The norm of stacked tensor is the norm used in gradient clipping.
        This is called global norm clipping.
    '''
    grads = [p.grad for p in parameters if p.grad is not None]
    norms = [torch.linalg.vector_norm(grad) for grad in grads]
    total_norm = torch.linalg.vector_norm(torch.stack(norms))

    clip_coef = max_l2_norm / (total_norm + eps)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for grad in grads:
        grad *= clip_coef_clamped

    return 

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
)-> tuple[torch.Tensor, torch.Tensor]:
    upper = dataset.shape[0] - context_length -1

    starts = np.random.randint(0, upper+1, size=batch_size) # (B,)
    offsets = np.arange(context_length) # (T,)
    idx = starts[:,None] + offsets[None,:] # (B,1) + (1,T) = (B,T)
    
    x_np = dataset[idx]
    y_np = dataset[idx+1]

    x = torch.from_numpy(x_np).to(device=device, dtype=torch.long)
    y = torch.from_numpy(y_np).to(device=device, dtype=torch.long)

    return x,y

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, 
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    obj = dict()
    obj['model_state'] = model.state_dict()
    obj['optimizer_state'] = optimizer.state_dict()
    obj['iteration'] = iteration

    torch.save(obj, out)

    return

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
                    model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    obj = torch.load(src)
    model.load_state_dict(obj['model_state'])
    optimizer.load_state_dict(obj['optimizer_state'])

    return obj['iteration']

def decoding(model: torch.nn.Module, prompt: torch.Tensor, eos_token_id:int, max_generate_tokens: int, 
             temperature: float=1.0, sampling_threshold: float=1.0, device: typing.Optional[torch.device] = None) -> torch.Tensor:
    '''
        prompt shape: (1,T) or (,T)
    '''
    if prompt.dim() == 1:
        prompt = prompt.unsqueeze(0) # (T,) -> (1,T)

    if device is None:
        device = next(model.parameters()).device()
    input = prompt.to(device)

    for _ in range(max_generate_tokens):
        logits = model(input)

        if logits.dim() == 3: # (1,T,V)
            next_logits = logits[0,-1,:]
        else: # (T,V)
            next_logits = logits[-1,:]

        # temperature-scaled logits
        if temperature == 0:
            idx = next_logits.argmax()
            probs = torch.zeros_like(next_logits)
            probs[idx] = 1.0
        else:
            probs = softmax(next_logits/temperature,-1)

        # top-p sampling (nucleus sampling)
        if sampling_threshold < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)

            remove = cumprobs > sampling_threshold
            remove[1:] = remove[:-1].clone()
            remove[0] = False

            remove_mask = torch.zeros_like(probs, device=probs.device,dtype=torch.bool).scatter(
                dim=-1, index=sorted_idx, src=remove
            )

            probs = probs.masked_fill(remove_mask, 0) # 将True的地方替换为0
            probs = probs / probs.sum(dim=-1)
    
        next_id = torch.multinomial(probs, num_samples=1)
        next_token = torch.tensor([[next_id]], device=device, dtype=input.dtype)

        input = torch.cat([input, next_token],dim=1)
        
        if next_id == eos_token_id:
            break

    return input.unsqueeze(0)