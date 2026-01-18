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

def _pretokenize_worker(args):
    m, merges = args
    token = [bytes([x]) for x in m]
    # print("token:",token)
    for merge in merges:
        i = 0
        while i < len(token)-1:
            # print(token[i],token[i+1],merge)
            if token[i] == merge[0] and token[i+1] == merge[1]:
                # print(merge)
                token[i] = merge[0] + merge[1]
                del token[i+1]
            i += 1
        if len(token) == 1:
            break
    return token

def parallel_tokenization_processes(text_segments, merges, ex):
    """
    Process a list of text segments in parallel.

    Args:
        text_segments: List of UTF-8 encoded byte strings to tokenize
        merges: BPE merge operations
        ex: ProcessPoolExecutor instance

    Returns:
        List of lists of tokens (one list per segment)
    """
    args = [(segment, merges) for segment in text_segments]
    results = list(ex.map(_pretokenize_worker, args))
    return results

class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.reverse_vocab = {v:k for k,v in vocab.items()}
        self.reverse_merges = map(lambda x:(self.reverse_vocab[x[0]],self.reverse_vocab[x[1]]),self.merges)
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
        # Setup regex patterns
        if self.special_tokens != None:
            split_pat = "(" + "|".join(map(re.escape, self.special_tokens)) + ")"
            split_re = re.compile(split_pat)
        else:
            split_re = re.compile(r'(?!)')

        pat_re = re.compile(PAT_STR)

        # Phase 1: Collect all segments to process
        pretokens = split_re.split(text)
        segments_to_process = []  # List of (type, data) tuples

        for pre_token in pretokens:
            if pre_token == "":
                continue
            elif self.special_tokens != None and pre_token in self.special_tokens:
                segments_to_process.append(('special', pre_token.encode('utf-8')))
            else:
                for m in pat_re.finditer(pre_token):
                    segments_to_process.append(('regular', m.group().encode('utf-8')))

        # Phase 2: Parallel process all regular segments
        regular_segments = [data for typ, data in segments_to_process if typ == 'regular']

        print(len(regular_segments))
        if regular_segments:
            nworkers = 10
            with ProcessPoolExecutor(max_workers=nworkers) as ex:
                tokenized_results = parallel_tokenization_processes(regular_segments, self.merges, ex)
        else:
            tokenized_results = []

        # Phase 3: Reconstruct in order
        encoded = []
        regular_idx = 0
        for typ, data in segments_to_process:
            if typ == 'special':
                encoded.append(data)
            else:  # regular
                encoded.extend(tokenized_results[regular_idx])
                regular_idx += 1

        # Convert bytes to token IDs
        return [self.reverse_vocab[x] for x in encoded]
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            yield from self.encode(s)
    
    def decode(self, ids: list[int]) -> str:
        s = b""
        for id in ids:
            s += self.vocab[id]        
        return s.decode('utf-8',errors='replace')


def _encode_iterable(tokenizer, iterable):
    """
    We place tokenizer.encode_iterable into a separate function so we can limit memory
    for just this function. We set the memory limit to 1MB.
    """
    yield from tokenizer.encode_iterable(iterable)

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

    # with open('./tokenizer_parameters/vocab_owt.dump','rb') as f:
    #     vocab = pickle.load(f)

    #     tokens = vocab.values()
    #     max_length_token = max(tokens,key=lambda x:len(x))
    #     print(max_length_token)
    #     count = 0
    #     for token in tokens:
    #         try:
    #             token.decode('utf-8')
    #         except UnicodeDecodeError:
    #             count += 1
        
    #     print(count)
    
    from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
    import tiktoken

    VOCAB_PATH = 'tests/fixtures/gpt2_vocab.json'
    MERGES_PATH = 'tests/fixtures/gpt2_merges.txt'
    FIXTURES_PATH = 'tests/fixtures/'

    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens = ['<|endoftext|>']
    )
    if b'<|endoftext|>' in tokenizer.vocab.values():
        print("111111")
    with open(FIXTURES_PATH + "tinystories_sample_5M.txt") as f:
        ids = []
        text = f.read()

        t1 = time.time()
        a = reference_tokenizer.encode(text,allowed_special={'<|endoftext|>'})
        t2 = time.time()
        print('tiktokenizer_time_used:',t2-t1)


        t1 = time.time()
        b = tokenizer.encode(text)
        t2 = time.time()
        print('tokenization_time_used:',t2-t1)
        assert a == b
        # for _id in _encode_iterable(tokenizer, f):
        #     # print(_id)
        #     ids.append(_id)
        
        # for _id in reference_tokenizer.encode_iterable(f):
        #     ids.append(_id)



    