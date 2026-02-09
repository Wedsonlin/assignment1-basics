from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import cross_entropy, learning_rate_schedule, gradient_clipping, get_batch, load_checkpoint, save_checkpoint

import torch
import argparse
import numpy as np
import time
import os

from torch.utils.tensorboard import SummaryWriter

def parse_tuple(s):
    return tuple(float(x) for x in s.split(','))

parser = argparse.ArgumentParser()

# model hyperparameters
parser.add_argument('--vocab_size', default=10000, type=int)
parser.add_argument('--context_length', default=256, type=int)
parser.add_argument('--d_model', default=512, type=int)
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--num_heads', default=16, type=int)
parser.add_argument('--d_ff', default=1344, type=int)
parser.add_argument('--rope_theta', default=100000, type=int)

# optimizer hyperparameters
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--weight_decay', default=0.01, type=float)
parser.add_argument('--betas', default="0.9,0.95", type=parse_tuple)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerLM(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    rope_theta=rope_theta
)
model = model.to(device)

optimizer = AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay,
    betas=betas,
    eps=eps
)

ckpt_load_path = ""
start_iter = 0
if os.path.exists(ckpt_load_path):
    start_iter = load_checkpoint(ckpt_load_path, model, optimizer)
    print(f"Resume from iter {start_iter}")

ckpt_save_dir = "G:\\cs336\\checkpoints\\"

train_dataset_path = "G:\\cs336\\parameters\\tokenID_tinystories_valid.npy"
valid_dataset_path = "G:\\cs336\\parameters\\tokenID_tinystories_valid.npy"

# ckpt_save_dir = "/home/lin/ckpts/"
# train_dataset_path = "/mnt/hgfs/parameters/tokenID_tinystories_valid.npy"
# valid_dataset_path = "/mnt/hgfs/parameters/tokenID_tinystories_valid.npy"

train_dataset = np.load(train_dataset_path)
valid_dataset = np.load(valid_dataset_path)
# train_dataset = np.memmap(train_dataset_path, dtype=np.uint16, mode='r')
# valid_dataset = np.memmap(valid_dataset_path, dtype=np.uint16, mode='r')

'''
    total tokens processed: 327,680,000
    total_tokens_processed = batch_size * num_step * context_length
'''
batch_size = 32
num_step = 1024


log_dir = "G:\\cs336\\log"
# log_dir = "/home/lin/cs336/log"
writer = SummaryWriter(log_dir=log_dir)

for step in range(start_iter,num_step):
    lr = learning_rate_schedule(step, lr_max=10, lr_min=6e-5,T_w=1000, T_c=10000)
    for group in optimizer.param_groups:
        group['lr'] = lr

    model.train()
    x, gt = get_batch(train_dataset, batch_size, context_length, device)
    logits = model(x)
    train_loss = cross_entropy(logits, gt)

    optimizer.zero_grad()
    train_loss.backward()
    gradient_clipping(model.parameters(), max_l2_norm=1.0)
    
    optimizer.step()

    if step % 1000 == 0:
        checkpoint_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        save_checkpoint(model,optimizer,step,out=ckpt_save_dir+f"ckpt_{step}_{checkpoint_time}")
        
    if step % 100 == 0 or step == num_step - 1:
        with torch.no_grad():
            model.eval()
            valid_x, valid_y = get_batch(valid_dataset, batch_size, context_length, device)
            valid_logits = model(valid_x)
            valid_loss = cross_entropy(valid_logits, valid_y)
            print(f"it:{step}, train_loss:{train_loss.item():.4f}, valid_loss:{valid_loss.item():.4f},lr:{lr}")

            writer.add_scalar("train_loss", train_loss, step)
            writer.add_scalar("valid_loss", valid_loss, step)
            writer.add_scalar("lr", lr, step)

save_checkpoint(model,optimizer,step,out=ckpt_save_dir+f"ckpt_final_{checkpoint_time}")
writer.close()
