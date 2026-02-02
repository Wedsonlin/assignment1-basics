from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import cross_entropy, learning_rate_schedule, gradient_clipping, get_batch, load_checkpoint, save_checkpoint

import torch
import argparse
import numpy as np

parser = argparse.ArgumentParser()

# model hyperparameters
parser.add_argument('--vocab_size', default=50257, type=int)
parser.add_argument('--context_length', default=1024, type=int)
parser.add_argument('--d_model', default=768, type=int)
parser.add_argument('--num_layers', default=12, type=int)
parser.add_argument('--num_heads', default=12, type=int)
parser.add_argument('--d_ff', default=3072, type=int)
parser.add_argument('--rope_theta', default=100000, type=int)

# optimizer hyperparameters
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--weight_decay', default=0.01, type=float)
parser.add_argument('--betas', default=(0.9, 0.95), type=tuple)
parser.add_argument('--eps', default=1e-8, type=float)


args = parser.parse_args()

vocab_size = args.vocab_size
context_length = args.context_length
d_model = args.d_model
num_layers = args.num_layers
num_heads = args.num_heads
d_ff = args.d_ff
rope_theta = args.rope_theta
lr = args.lr
weight_decay = args.weight_decay
betas = args.betas
eps = args.eps

print("-"*50,"model hyperparameters","-"*50)
print("vocab_size:", vocab_size)
print("context_length:", context_length)
print("d_model:", d_model)
print("num_layers:", num_layers)
print("num_heads:", num_heads)
print("d_ff:", d_ff)
print("rope_theta:", rope_theta)

print("-"*50,"optimizer hyperparameters","-"*50)
print("lr:", lr)
print("weight_decay:", weight_decay)
print("betas:", betas)
print("eps:", eps)


model = TransformerLM(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    rope_theta=rope_theta
)

optimizer = AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay,
    betas=betas,
    eps=eps
)

checkpoint_path = ""
iteration = None
try:
    iteration = load_checkpoint(checkpoint_path, model, optimizer)
except FileNotFoundError:
    pass



dataset_path = "G:\\cs336\\parameters\\tokenID_tinystories_train.npy"
dataset = np.load(dataset_path)
# dataset = np.memmap(dataset_path)

'''
    total tokens processed: 327,680,000
    total_tokens_processed = batch_size * num_step * context_length
'''
batch_size = 32
num_step = 40000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if iteration is not None:
    num_step -= iteration

for step in range(num_step):
    x, gt = get_batch(dataset, batch_size, context_length, device)
    y = model(x)
    loss = cross_entropy(y, gt)
    loss.backward()
    gradient_clipping(model.parameters, max_l2_norm=10)

    lr = learning_rate_schedule(step, lr_max=10, lr_min=1, T_w=100, T_c=1000)
    for group in optimizer.param_groups:
        group['lr'] = lr
    
    optimizer.step()



