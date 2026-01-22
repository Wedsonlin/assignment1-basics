from cs336_basics.pretokenization import chunking,parallel_pretokenization_processes
from collections import defaultdict
from collections.abc import Iterator, Iterable
import time
import pickle
import regex as re
from concurrent.futures import ProcessPoolExecutor

PAT_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path,vocab_size=1000,special_tokens=['<|endoftext|>']):

    vocab : dict[int, bytes] = {x:bytes([x]) for x in range(256)}
    merges : list[tuple[bytes,bytes]] = []


    for i in range(len(special_tokens)):
        vocab[256+i] = special_tokens[i].encode('utf-8') 

    num_merges = vocab_size - 256 - len(special_tokens) # 0-255 和 special_tokens

    num_of_process = 18

    print('start chunking')
    t1 = time.time()
    chunks = chunking(input_path, num_of_chunk=num_of_process) # 对文件进行分块
    t2 = time.time()
    print('end chunking')
    print('chukning_time_used:',t2-t1)

    counter = defaultdict(int)
    print('start pretokenization')
    t1 = time.time()
    counter = parallel_pretokenization_processes(chunks, special_tokens, num_of_process)
    t2 = time.time()
    
    print('end pretokenization')
    print('pretokenization_time_used:',t2-t1)

    pair_frequency = defaultdict(int)

    for x in counter.keys():
        for bigram in zip(x,x[1:]):
            pair_frequency[bigram] += counter[x]

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
        for x,freq in counter.items():
            i = 0
            if index1 in x and index2 in x:
                x_copy = x
                while i < len(x_copy):
                    if i+1 < len(x_copy) and x_copy[i] == index1 and x_copy[i+1] == index2:
                        pair_frequency[(x_copy[i],x_copy[i+1])] -= counter[x]
                        if i-1 >= 0:
                            pair_frequency[(x_copy[i-1],x_copy[i])] -= counter[x]
                            pair_frequency[(x_copy[i-1],new_index)] += counter[x]
                        if i+2 < len(x_copy):
                            pair_frequency[(x_copy[i+1],x_copy[i+2])] -= counter[x]
                            pair_frequency[(new_index,x_copy[i+2])] += counter[x]
                        x_copy = x_copy[:i] + (new_index,) + x_copy[i+2:]
                    i += 1
                words_to_update[(x,x_copy)] = freq
        for k,v in words_to_update.items():
            del counter[k[0]]
            counter[k[1]] = v

    t2 = time.time()
    print('end merging')
    print('merging_time_used:',t2-t1)
    return vocab,merges 

def split_by_special(text,special_tokens,drop_special=False):
    if special_tokens == None:
        return [text]
    split_pat = "|".join(map(re.escape, special_tokens))
    if not drop_special:
        split_pat = f"({split_pat})"
    split_re = re.compile(split_pat)
    return [c for c in split_re.split(text) if c]

def merge_and_encode(chunk,merges,reverse_vocab):
    pat_re = re.compile(PAT_STR)
    tokens = []
    for pretoken in pat_re.findall(chunk): 
        token = [bytes([x]) for x in pretoken.encode('utf-8')]
        while True:
            i = 0
            merge_pos = -1
            min_id = len(reverse_vocab)
            while i < len(token)-1:
                pair = (token[i],token[i+1])
                if pair in merges:
                    token_id = reverse_vocab.get(token[i]+token[i+1])
                    if token_id is not None and token_id < min_id: # merge后的token_id越小，说明该merge越靠前
                        min_id = token_id
                        merge_pos = i
                i += 1
            if merge_pos == -1:
                break
            token = token[:merge_pos] + [token[merge_pos]+token[merge_pos+1]] + token[merge_pos+2:]
        tokens.extend(token)
    return [reverse_vocab[x] for x in tokens]

class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = set(merges) # 底层是哈希表，查找的时间复杂度为O(1)
        self.reverse_vocab = {v:k for k,v in vocab.items()}
        self.special_tokens = special_tokens
        if special_tokens != None:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
    
    @classmethod
    def from_files(cls,vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath,'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath,'rb') as f:
            merges = pickle.load(f)
        
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str)  -> list[int] :
        if text == "":
            return []
        chunks = split_by_special(text,self.special_tokens,drop_special=False)
        tokens = []
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append(self.reverse_vocab[chunk.encode('utf-8')])
            else:
                tokens.extend(merge_and_encode(chunk,self.merges,self.reverse_vocab))
        
        return tokens
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            yield from self.encode(s)
    
    def decode(self, ids: list[int]) -> str:
        s = b""
        for id in ids:
            s += self.vocab[id]        
        return s.decode('utf-8',errors='replace')

if __name__ == "__main__":
    # input_path = "./tests/fixtures/corpus.en"
    # start_time = time.time()
    # vocab, merges = train_bpe(
    #     input_path=input_path,
    #     vocab_size=500,
    #     special_tokens=["<|endoftext|>"],
    # )
    # end_time = time.time()


    # input_path = "G:\\cs336\\data\\owt_train.txt"
    # start_time = time.time()
    # vocab, merges = train_bpe(
    #     input_path=input_path,
    #     vocab_size=32000,
    #     special_tokens=["<|endoftext|>"],
    # )
    # end_time = time.time()
    # print("total_time_used:",end_time - start_time)
    # with open('vocab_owt.dump','wb') as f:
    #     pickle.dump(vocab,f)
    # with open('merges_owt.dump','wb') as f:
    #     pickle.dump(merges,f)
    

    VOCAB_PATH = 'G:/cs336/parameters/vocab_tinystories.dump'
    MERGES_PATH = 'G:/cs336/parameters/merges_tinystories.dump'
    special_tokens=['<|endoftext|>']
    tokenizer = Tokenizer.from_files(VOCAB_PATH,MERGES_PATH,special_tokens)

    import numpy as np
    tokens = []
    with open("G:/cs336/data/" + "owt_train.txt",'r',encoding='utf-8') as f:
        t1 = time.time()
        for token in tokenizer.encode_iterable(f):
            tokens.append(token)
        t2 = time.time()
        sec = t2-t1
        print("time_used:",sec)

        # total_bytes = len(text.encode('utf-8'))
        # t1 = time.time()
        # tokens = tokenizer.encode(text)
        # t2 = time.time()
        # sec = t2-t1
        # print(sec)
        # throughput = total_bytes / sec
        # print("throughput:",throughput)

        tokens = np.array(tokens,dtype=np.uint16)
        np.save("G:/cs336/parameters/tokenID_owt_train.npy",tokens)
    
    result = np.load("G:/cs336/parameters/tokenID_owt_train.npy")
    print(result)
    print(len(result))
    print(type(result))
    print(result.dtype)
    # with open("G:/cs336/parameters/tokenID_tinystories_train.dump","rb") as f:
    #     tokens = pickle.load(f)
    #     tokens = np.array(tokens,dtype=np.uint16)
    #     np.save("G:/cs336/parameters/tokenID_tinystories_train.npy",tokens)
