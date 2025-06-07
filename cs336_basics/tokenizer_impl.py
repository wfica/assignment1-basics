from collections.abc import Iterable
from collections import Counter
import regex as re
import heapq
from collections import defaultdict
import multiprocessing
from typing import BinaryIO
import os

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class ComparableItem:
    def __init__(self, cnt, idx_1, idx_2, vocab):
        self.data = (cnt, idx_1, idx_2)
        self.integer_val = cnt
        self.string_val1 = vocab[idx_1]
        self.string_val2 = vocab[idx_2]

    # Invert the comparison logic to create a max-heap
    def __lt__(self, other):
        if self.integer_val != other.integer_val:
            return self.integer_val > other.integer_val
    
        if self.string_val1 != other.string_val1:
            return self.string_val1 > other.string_val1
      
        return self.string_val2 > other.string_val2

    def __repr__(self):
        return f"Item({self.data})"
    

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

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

def pre_tokenize_counter(text: str, special_tokens=["<|endoftext|>"]) -> dict[tuple[int], int]:
  
  pattern = '|'.join(re.escape(token) for token in special_tokens)
  def _get_pre_tokens_as_bytes_tuple(text: str) -> Iterable[list[int]]:
    for chunk in re.split(pattern, text):
      for matched_pre_token in re.finditer(PAT, chunk):
        yield tuple(byte for byte in matched_pre_token.group(0).encode('utf-8'))


  return Counter(_get_pre_tokens_as_bytes_tuple(text))

def merge_frq_tbls(frq_tbls: list[dict[tuple[int], int]]) -> list[list[list[int], int]]:
  union = frq_tbls[0]
  for counter in frq_tbls[1:]:
    union.update(counter)
  
  ret = []
  for bytes_in_word, cnt in union.items():
    ret.append([list(bytes_in_word), cnt])
  return ret


def pre_tokenize(text: str, special_tokens=["<|endoftext|>"]) -> list[list[list[int], int]]:
  ret = []
  for bytes_in_word, cnt in pre_tokenize_counter(text, special_tokens).items():
    ret.append([list(bytes_in_word), cnt])
  return ret

def get_inital_vocab(special_tokens=["<|endoftext|>"]) -> dict[int, bytes]:
  vocab: dict[bytes, int] = {i: bytes([i]) for i in range(0, 256)}
  for special_token in special_tokens:
    vocab[len(vocab)] = special_token.encode("utf-8")
  return vocab


def get_pairs_cnts(frq_tbl):
  pairs_to_word_idx: dict[tuple[int, int], list[int]] = defaultdict(list)
  pairs_cnts = Counter()
  for word_idx, (idx_tuple, cnt) in enumerate(frq_tbl):
    for i in range(0, len(idx_tuple)-1):
      pair = (idx_tuple[i], idx_tuple[i+1])
      pairs_cnts[pair] += cnt
      pairs_to_word_idx[pair].append(word_idx)
  return pairs_to_word_idx, pairs_cnts

def pair_idx_to_pair_of_strings(p, vocab):
  return (vocab[p[0]], vocab[p[1]])


def create_frq_tbl_from_file(file_path: str, special_tokens:list[str], num_processes)-> list[list[list[int], int]]:
  assert len(special_tokens) == 1 or num_processes == 1
  if num_processes == 1:
    with open(file_path, "r") as f:
      txt = f.read()
      return pre_tokenize(txt, special_tokens)
  with open(file_path, "rb") as f:
    assert len(special_tokens) == 1
    special_token = special_tokens[0]
    boundaries = find_chunk_boundaries(f, num_processes, special_token.encode("utf-8"))

    pool = multiprocessing.Pool(processes=num_processes)
    results = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        results.append(pool.apply_async(pre_tokenize_counter, [chunk, [special_token]]))

    counters = [r.get() for r in results]
    return merge_frq_tbls(counters)

def find_max_element(max_heap: list[ComparableItem], pairs_cnts: dict[tuple[int, int], int]):
  while len(max_heap) > 0:
    max_elem = heapq.heappop(max_heap)
    # Check if the max_elem is still valid
    if max_elem.data[0] == pairs_cnts[(max_elem.data[1], max_elem.data[2])]:
      return max_elem.data[1], max_elem.data[2]  # Return the pair of indices
  return None

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], debug_text=None, num_processes=32):
  """
  Args:
    input_path: str Path to a text file with BPE tokenizer training data.
    vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
    initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
    special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
      otherwise affect BPE training.

  Returns:
    vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
    merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
      is a tuple of bytes (<tokens1>, <tokens2>), representing that <tokens1> were merged with
      <tokens2>. The merges should be ordered by order of creation.
  """
  assert input_path is not None or debug_text is not None

  frq_tbl = create_frq_tbl_from_file(input_path, special_tokens, num_processes) if input_path is not None else pre_tokenize(debug_text, special_tokens)
  vocab = get_inital_vocab(special_tokens)
  merges: list[tuple[bytes, bytes]] = []

  # Get the initial counts of pairs and a mapping between pairs and word_idx
  pairs_to_word_idx, pairs_cnts = get_pairs_cnts(frq_tbl)
  max_heap = []
  for pair, cnt in pairs_cnts.items():
    max_heap.append(ComparableItem(cnt, pair[0], pair[1], vocab))
  heapq.heapify(max_heap)

  # Start merges
  while len(vocab) < vocab_size:
    # find the most common pair, first order by occurrences counts, resolve ties by lexicographic order
    most_frq_pair = find_max_element(max_heap, pairs_cnts)
    if most_frq_pair is None:
      break  # No more pairs to merge
    merges.append((vocab[most_frq_pair[0]], vocab[most_frq_pair[1]]))
    # remove it from the dict
    pairs_cnts[most_frq_pair] = 0
    # add it to the vocab
    new_token = len(vocab)
    vocab[new_token] = vocab[most_frq_pair[0]] +  vocab[most_frq_pair[1]]
    # update all the affected pairs counts
    for word_idx in pairs_to_word_idx[most_frq_pair]:
      word_cnt = frq_tbl[word_idx][1]
      old_indexes = frq_tbl[word_idx][0]
      new_indexes = []
      i = 0
      while i < len(old_indexes):
        old_token = old_indexes[i]
        if i + 1 < len(old_indexes) and old_indexes[i] == most_frq_pair[0] and old_indexes[i+1] == most_frq_pair[1]:
          new_indexes.append(new_token)
          # update the previous pair
          if i > 0:
            pairs_cnts[(old_indexes[i-1], old_indexes[i])] -= word_cnt
            heapq.heappush(max_heap, ComparableItem(pairs_cnts[(old_indexes[i-1], old_indexes[i])], old_indexes[i-1], old_indexes[i], vocab))
            pairs_cnts[(old_indexes[i-1], new_token)] += word_cnt
            heapq.heappush(max_heap, ComparableItem(pairs_cnts[(old_indexes[i-1], new_token)], old_indexes[i-1], new_token, vocab))
            pairs_to_word_idx[(old_indexes[i-1], new_token)].append(word_idx)
          # update the following pair
          if i < len(old_indexes)-2:
            pairs_cnts[(old_indexes[i+1], old_indexes[i+2])] -= word_cnt
            heapq.heappush(max_heap, ComparableItem(pairs_cnts[(old_indexes[i+1], old_indexes[i+2])], old_indexes[i+1], old_indexes[i+2], vocab))
            pairs_cnts[(new_token, old_indexes[i+2])] += word_cnt
            heapq.heappush(max_heap, ComparableItem(pairs_cnts[(new_token, old_indexes[i+2])], new_token, old_indexes[i+2], vocab))
            pairs_to_word_idx[(new_token, old_indexes[i+2])].append(word_idx)
          i += 2
        else:
          new_indexes.append(old_token)
          i += 1
      frq_tbl[word_idx] = (new_indexes, word_cnt)

  return vocab, merges