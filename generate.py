import torch

from cs336_basics.utils import load_checkpoint
from cs336_basics.utils import decoding
from cs336_basics.model import TransformerLM
from cs336_basics.bpe import Tokenizer

vocab_size = 10000
context_length = 256
d_model = 512
num_layers = 4
num_heads = 16
d_ff = 1344
rope_theta = 100000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_load_path = "G:\\cs336\\checkpoints\\ckpt_final_2026-03-03-21-02"

def generate(model, prompt, device, max_generate_tokens=256, temperature: float=1.0, sampling_threshold: float=1.0):
    model = model.to(device)

    VOCAB_PATH = 'G:/cs336/parameters/vocab_tinystories.dump'
    MERGES_PATH = 'G:/cs336/parameters/merges_tinystories.dump'
    special_tokens=['<|endoftext|>']
    tokenizer = Tokenizer.from_files(VOCAB_PATH,MERGES_PATH,special_tokens)
    input_tokens = tokenizer.encode(prompt)
    input_tokens = torch.tensor(input_tokens, device=device)

    print(f"input_tokens:{len(input_tokens)}")
    output_tokens = decoding(model, 
                             input_tokens, 
                             eos_token_id=256, 
                             max_generate_tokens=max_generate_tokens, 
                             temperature=temperature, 
                             sampling_threshold=sampling_threshold, 
                             device=device)
    
    output_tokens = output_tokens.tolist()
    print(f"output_tokens:{len(output_tokens)}")
    return tokenizer.decode(output_tokens)

if __name__ == "__main__":
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    )

    load_checkpoint(ckpt_load_path, model, None)

    prompt = "Once upon a time, there was a little girl named Lily."
    result = generate(model, prompt, device, max_generate_tokens=1000, temperature=0.6, sampling_threshold=0.95)

    print(result)