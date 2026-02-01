import torch
from collections.abc import Callable, Iterable
from typing import Optional 
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get('t', 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t+1) * grad
                state['t'] = t + 1
        
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.95), eps=1e-8):
        defaults = {
            "alpha": lr,
            "betas": betas,
            "eps": eps,
            "lambda": weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable]=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            alpha = group['alpha']
            betas = group['betas']
            eps = group['eps']
            Lamda = group['lambda']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get('t', 1)
                m = state.get('m', torch.zeros_like(p))
                v = state.get('v', torch.zeros_like(p))

                grad = p.grad.data
                m = betas[0]*m + (1-betas[0])*grad
                v = betas[1]*v + (1-betas[1])*torch.pow(grad,2)
                alpha_t = alpha*math.sqrt(1-math.pow(betas[1],t))/(1-math.pow(betas[0],t))
                p.data -= alpha_t*m/(torch.sqrt(v)+eps)
                p.data -= alpha*Lamda*p.data

                state['t'] = t + 1
                state['m'] = m
                state['v'] = v

        return loss
