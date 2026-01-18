from cs336_basics.pretokenization import chunking,parallel_pretokenization_processes
from collections import defaultdict, Iterable, Iterable
import time
import pickle
import regex as re

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

class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.reverse_vocab = {v:k for k,v in vocab.items()}
        self.reverse_merges = map(lambda x:(self.reverse_vocab[x[0]],self.reverse_vocab[x[1]]),self.merges)
    
    @classmethod
    def from_files(cls,vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath,'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath,'rb') as f:
            merges = pickle.load(f)
        
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str)  -> list[int] :
        PAT_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pat_re = re.compile(PAT_STR)

        split_pat = "|".join(map(re.escape, self.special_tokens))
        split_re = re.compile(split_pat)
        indices = [match.start() for match in split_re.finditer(text)]

        sc_token_seq = []
        for index in indices:
            for sc_token in self.special_tokens:
                if text[index:index+len(sc_token)] == sc_token:
                    sc_token_seq.append(self.reverse_vocab[sc_token.encode('utf-8')])

        encoded = []
        for chunk in split_re.split(text):
            tokens = []
            for m in pat_re.finditer(chunk):
                token = [chr(b).encode('utf-8') for b in m.group().encode('utf-8')]
                for merge in self.merges:
                    i = 0
                    while i < len(token)-1:
                        if token[i] == merge[0] and token[i+1] == merge[1]:
                            token[i] = merge[0] + merge[1]
                            del token[i+1]
                        i += 1
                    if len(token) == 1:
                        break
                
                tokens += token
            encoded.append(tokens)

        encoded_text = []
        i = 0
        for x in encoded:
            for b in x:
                encoded_text.append(self.reverse_vocab[b])
            encoded_text.append(sc_token_seq[i])
            i += 1
        
        if i < len(sc_token_seq):
            encoded_text.append(sc_token_seq[i])

        return encoded_text
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        
        return
    
    def decode(self, ids: list[int]) -> str:
        s = ""
        for id in ids:
            s += self.vocab[id]
        return s 


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

    with open('./tokenizer_parameters/vocab_tinystories.dump','rb') as f:
        vocab = pickle.load(f)

        tokens = vocab.values()
        max_length_token = max(tokens,key=lambda x:len(x))
        print(max_length_token)