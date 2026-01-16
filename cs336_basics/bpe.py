from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import time
import pickle

def build_regex(special_tokens: list[str]):
    PAT_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    split_pat = "|".join(map(re.escape, special_tokens))
    return re.compile(split_pat), re.compile(PAT_STR)

def pretokenize_chunk(chunk: str, split_re, pat_re) -> defaultdict:
    d = defaultdict(int)
    for sc in split_re.split(chunk):
        for m in pat_re.finditer(sc):
            token = tuple(map(int,m.group().encode('utf-8')))
            # token = tuple(m.group().encode('utf-8'))
            d[token] += 1
    return d

def parallel_pretokenization_threads_per_chunk(chunks, special_tokens, nthreads):
    split_re, pat_re = build_regex(special_tokens)

    total = defaultdict(int)
    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        for d in ex.map(lambda c: pretokenize_chunk(c, split_re, pat_re), chunks):
            for k, v in d.items():
                total[k] += v
    return total

def train_bpe(input_path,vocab_size=1000,special_tokens=['<|endoftext|>']):

    vocab : dict[int, bytes] = {x:bytes([x]) for x in range(256)}
    merges : list[tuple[bytes,bytes]] = []


    for i in range(len(special_tokens)):
        vocab[256+i] = special_tokens[i].encode('utf-8') 

    num_merges = vocab_size - 256 - len(special_tokens) # 0-255 和 special_tokens

    num_threads = 10

    print('start chunking')
    t1 = time.time()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_threads, b"<|endoftext|>")
        
        d = defaultdict(int)
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore") # 分块
            chunk = re.sub(r"\r\n?", "\n", chunk) # 统一换行符
            chunks.append(chunk)
    t2 = time.time()
    print('end chunking')
    print('chukning_time_used:',t2-t1)

    print('start pretokenization')
    t1 = time.time()
    d = parallel_pretokenization_threads_per_chunk(chunks,special_tokens,num_threads)
    t2 = time.time()
    print('end pretokenization')
    print('pretokenization_time_used:',t2-t1)

    pair_frequency = defaultdict(int)

    for x in d.keys():
        for bigram in zip(x,x[1:]):
            pair_frequency[bigram] += d[x]

    print('start merging')
    t1 = time.time()
    for count in range(num_merges):
        '''
            不是 lexicographical order, 因为很可能解码不出字符，实际上是 byte order
        '''
        pair = max(pair_frequency.items(), key=lambda x:(x[1],vocab[x[0][0]],vocab[x[0][1]]))[0]
        index1, index2 = pair
        new_index = 256 + len(special_tokens) + count
        vocab[new_index] = vocab[index1] + vocab[index2]
        merges.append((vocab[index1],vocab[index2]))
        
        # merge
        words_to_update = defaultdict(int)
        for x,freq in d.items():
            i = 0
            if index1 in x and index2 in x:
                x_copy = x
                while i < len(x_copy):
                    if i+1 < len(x_copy) and x_copy[i] == index1 and x_copy[i+1] == index2:
                        pair_frequency[(x_copy[i],x_copy[i+1])] -= d[x]
                        if i-1 >= 0:
                            pair_frequency[(x_copy[i-1],x_copy[i])] -= d[x]
                            pair_frequency[(x_copy[i-1],new_index)] += d[x]
                        if i+2 < len(x_copy):
                            pair_frequency[(x_copy[i+1],x_copy[i+2])] -= d[x]
                            pair_frequency[(new_index,x_copy[i+2])] += d[x]
                        x_copy = x_copy[:i] + (new_index,) + x_copy[i+2:]
                    i += 1
                words_to_update[(x,x_copy)] = freq
        for k,v in words_to_update.items():
            del d[k[0]]
            d[k[1]] = v

    t2 = time.time()
    print('end merging')
    print('merging_time_used:',t2-t1)
    return vocab,merges 

if __name__ == "__main__":
    input_path = "./tests/fixtures/corpus.en"
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()


    # input_path = "D:\\cs336\\data\\TinyStoriesV2-GPT4-train.txt"
    # start_time = time.time()
    # vocab, merges = train_bpe(
    #     input_path=input_path,
    #     vocab_size=10000,
    #     special_tokens=["<|endoftext|>"],
    # )
    # end_time = time.time()
    print("total_time_used:",end_time - start_time)
    with open('vocab.dump','wb') as f:
        pickle.dump(vocab,f)
    with open('merges.dump','wb') as f:
        pickle.dump(merges,f)

    # with open('voacb.txt','rb') as f:
    #     vocab = pickle.load(f)
    #     print(vocab)