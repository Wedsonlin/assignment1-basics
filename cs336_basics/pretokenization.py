import os
import regex as re
from typing import BinaryIO
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def chunking(input_path, num_of_chunk):
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_of_chunk, b"<|endoftext|>")
        
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore") # 分块
            chunk = re.sub(r"\r\n?", "\n", chunk) # 统一换行符
            chunks.append(chunk)



    return chunks

# 全局正则表达式模式字符串（用于多进程中重新编译）
PAT_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _pretokenize_chunk_worker(args):
    """Worker 函数，在每个进程中重新编译正则表达式"""
    chunk, special_tokens = args
    split_pat = "|".join(map(re.escape, special_tokens))
    split_re = re.compile(split_pat)
    pat_re = re.compile(PAT_STR)

    d = defaultdict(int)
    for sc in split_re.split(chunk):
        for m in pat_re.finditer(sc):
            token = tuple(map(int, m.group().encode('utf-8')))
            d[token] += 1
    return dict(d)  # 返回普通 dict，便于序列化

def parallel_pretokenization_processes(chunks, special_tokens, nworkers):
    """使用多进程并行 pretokenization，绕过 GIL 限制"""
    total = defaultdict(int)
    args = [(chunk, special_tokens) for chunk in chunks]

    with ProcessPoolExecutor(max_workers=nworkers) as ex:
        for d in ex.map(_pretokenize_chunk_worker, args):
            for k, v in d.items():
                total[k] += v
    return total



## Usage
# with open(..., "rb") as f:
#     num_processes = 4
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

#     # The following is a serial implementation, but you can parallelize this
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token
