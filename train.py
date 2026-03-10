from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import cross_entropy, learning_rate_schedule, gradient_clipping, get_batch, load_checkpoint, save_checkpoint

import torch
import argparse
import numpy as np
import time
import os

from torch.utils.tensorboard import SummaryWriter
import wandb

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
parser.add_argument('--rope_theta', default=10000, type=int)

# optimizer hyperparameters
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--weight_decay', default=0.01, type=float)
parser.add_argument('--betas', default="0.9,0.99", type=parse_tuple)
parser.add_argument('--eps', default=1e-8, type=float)


args = parser.parse_args()

vocab_size = 32000
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

ckpt_save_dir = "/data/coding/checkpoints/"
train_dataset_path = "/data/coding/data/tokenID_owt_train.npy"
valid_dataset_path = "/data/coding/data/tokenID_owt_valid.npy"

# train_dataset = np.load(train_dataset_path)
# valid_dataset = np.load(valid_dataset_path)
train_dataset = np.memmap(train_dataset_path, dtype=np.uint16, mode='r')
valid_dataset = np.memmap(valid_dataset_path, dtype=np.uint16, mode='r')

'''
    total tokens processed: 327,680,000
    total_tokens_processed = batch_size * num_step * context_length
'''
context_length = 256
batch_size = 64
num_step = 20000


log_dir = "/data/coding/logs"
writer = SummaryWriter(log_dir=log_dir)

model = TransformerLM(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    rope_theta=rope_theta
).to(device)

optimizer = AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay,
    betas=betas,
    eps=eps
)

start_iter = 0

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="ltao02845-sun-yat-sen-university",
    # Set the wandb project where this run will be logged.
    project="train-on-owt",
    # Track hyperparameters and run metadata.
    # config={
    #     "learning_rate": lr
    # },
    # name=f"lr-{lr}",
    # resume="allow"
)

for step in range(start_iter,num_step):
    step_lr = learning_rate_schedule(step, lr_max=lr, lr_min=lr*0.1,T_w=num_step//10, T_c=num_step)
    for group in optimizer.param_groups:
        group['lr'] = step_lr

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
            print(f"it:{step}, train_loss:{train_loss.item():.4f}, valid_loss:{valid_loss.item():.4f},lr:{step_lr}")

            run.log({
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "lr": step_lr
            }, step=step)
            writer.add_scalar("train_loss", train_loss, step)
            writer.add_scalar("valid_loss", valid_loss, step)
            writer.add_scalar("lr", step_lr, step)

save_checkpoint(model,optimizer,step,out=ckpt_save_dir+f"ckpt_final_{checkpoint_time}")
run.finish()

writer.close()
