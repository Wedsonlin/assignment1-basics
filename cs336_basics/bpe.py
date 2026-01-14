from pretokenization_example import find_chunk_boundaries
import regex as re
from collections import defaultdict
def train_bpe(input_path,vocab_size=1000,special_tokens=['<|endoftext|>']):

    vocab : dict[int, bytes] = {x:bytes([x]) for x in range(256)}
    merges : list[tuple[bytes,bytes]] = []

    num_merges = 2

    num_processes = 4
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            
            pattern = '|'.join(map(re.escape,special_tokens))
            splited_chunk = re.split(pattern,chunk)

            d = defaultdict(int)
            for sc in splited_chunk:
                pre_tokens = re.finditer(PAT,sc)

                for x in pre_tokens:
                    s = list(map(int,x.group().encode('utf-8')))
                    d[s] += 1

                for i in range(num_merges):
                    adjacent_frequency = defaultdict(int)
                    for x in d.keys():
                        for bigram in zip(x,x[1:]):
                            adjacent_frequency[bigram] += d[x]

                    
                    adjacent_frequency = sorted(adjacent_frequency.items(),key=lambda x:x[1],reverse=True)
                    index1, index2 = adjacent_frequency[0][0]
                    new_index = 256 + i
                    vocab[new_index] = vocab[index1] + vocab[index2]

                    for x in d.keys():
                        new_key = []
                        i = 0
                        while i < len(x):
                            if i+1 < len(x) and x[i] == index1 and x[i+1] == index2:
                                new_key.append(new_index)
                                i += 2
                            else:
                                new_key.append(index1)
                                i += 1
                        d[new_key] = d.pop(x)

                                

                
                print(vocab)
                print(merges)
                    

                break
            
            print(d)
            print(adjacent_frequency)

            break



input_path = "../../data/TinyStoriesV2-GPT4-valid.txt"
vocab_size = 1000
special_tokens = ['<|endoftext|>']

train_bpe(input_path,vocab_size,special_tokens)

