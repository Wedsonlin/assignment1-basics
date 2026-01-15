from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import defaultdict
import multiprocessing
from tqdm import tqdm

def _init_worker(special_tokens: list[str]):
    global _SPLIT_RE, _PAT_RE
    split_pat = "|".join(map(re.escape, special_tokens))
    _SPLIT_RE = re.compile(split_pat)

    PAT_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    _PAT_RE = re.compile(PAT_STR)


def pretokenization(chunk:str):
    d = defaultdict(int)
    for sc in _SPLIT_RE.split(chunk):
        for x in _PAT_RE.finditer(sc): # regex-based pre-tokenizer
            s = tuple(map(int,x.group().encode('utf-8')))
            d[s] += 1
    return d

def parallel_pretokenization(
    chunks: list[str],
    special_tokens: list[str],
    nproc: int | None = None,
    chunksize: int = 1,
) -> defaultdict:
    total = defaultdict(int)
    with multiprocessing.Pool(processes=nproc, initializer=_init_worker, initargs=(special_tokens,)) as pool:
        for d in pool.imap_unordered(pretokenization, chunks, chunksize=chunksize):
            # 聚合每个 worker 的 defaultdict
            for k, v in d.items():
                total[k] += v

    return total 


def train_bpe(input_path,vocab_size=1000,special_tokens=['<|endoftext|>']):

    vocab : dict[int, bytes] = {x:bytes([x]) for x in range(256)}
    merges : list[tuple[bytes,bytes]] = []


    for i in range(len(special_tokens)):
        vocab[256+i] = special_tokens[i].encode('utf-8') 

    num_merges = vocab_size - 256 - len(special_tokens) # 0-255 和 special_tokens

    num_processes = 6

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        
        d = defaultdict(int)
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore") # 分块
            chunk = re.sub(r"\r\n?", "\n", chunk) # 统一换行符
            chunks.append(chunk)

    d = parallel_pretokenization(chunks,special_tokens,nproc=num_processes,chunksize=1)

    adjacent_frequency = defaultdict(int)
    for x in tqdm(d.keys()):
        for bigram in zip(x,x[1:]):
            adjacent_frequency[bigram] += d[x]

    for count in range(num_merges):
        '''
            排序不是 lexicographical order, 因为很可能解码不出字符，实际上是 byte order
        '''
        frequency_list = sorted(adjacent_frequency.items(),key=lambda x:(x[1],vocab[x[0][0]],vocab[x[0][1]]),reverse=True)

        index1, index2 = frequency_list[0][0]
        new_index = 256 + len(special_tokens) + count
        vocab[new_index] = vocab[index1] + vocab[index2]
        merges.append((vocab[index1],vocab[index2]))
        
        # merge
        new_d = defaultdict(int)
        for x,freq in d.items():
            i = 0
            x_copy = x
            while i < len(x_copy):
                if i+1 < len(x_copy) and x_copy[i] == index1 and x_copy[i+1] == index2:
                    adjacent_frequency[(x_copy[i],x_copy[i+1])] -= d[x]
                    if i-1 >= 0:
                        adjacent_frequency[(x_copy[i-1],x_copy[i])] -= d[x]
                        adjacent_frequency[(x_copy[i-1],new_index)] += d[x]
                    if i+2 < len(x_copy):
                        adjacent_frequency[(x_copy[i+1],x_copy[i+2])] -= d[x]
                        adjacent_frequency[(new_index,x_copy[i+2])] += d[x]
                    x_copy = x_copy[:i] + (new_index,) + x_copy[i+2:]
                i += 1
            # new_d.append((tuple(new_key),d[x]))
            new_d[x_copy] = freq
        # d = defaultdict(int,new_d)
        d = new_d
    return vocab,merges 

import time
import json
import pickle
if __name__ == "__main__":
    input_path = "./tests/fixtures/corpus.en"
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()


    # input_path = "G:\\cs336\\data\\TinyStoriesV2-GPT4-train.txt"
    # start_time = time.time()
    # vocab, merges = train_bpe(
    #     input_path=input_path,
    #     vocab_size=10000,
    #     special_tokens=["<|endoftext|>"],
    # )
    # end_time = time.time()
    print("time_used:",end_time - start_time)
    with open('1.txt','wb') as f:
        pickle.dump(vocab,f)
        pickle.dump(merges,f)
    # assert (end_time - start_time) < 1.5