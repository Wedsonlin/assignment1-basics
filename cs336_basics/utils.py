import torch
from jaxtyping import Float, Int
from math import cos, pi
from collections.abc import Iterable

def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    max_element = x.amax(dim=i,keepdim=True) # avoid overflow
    exp_y = torch.exp(x-max_element)
    return exp_y / exp_y.sum(dim=i,keepdim=True)

def cross_entropy(logits: Float[torch.Tensor, " batch_size vocab_size"], targets: Int[torch.Tensor, " batch_size"]) -> Float[torch.Tensor, ""]:
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
