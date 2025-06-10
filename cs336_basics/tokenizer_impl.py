from collections.abc import Iterable
from collections import Counter
import regex as re
import heapq
from collections import defaultdict
import multiprocessing
from typing import BinaryIO
import os
import pickle

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

def pre_tokenize(text: str, special_tokens=["<|endoftext|>"], special_token_to_index:dict[str, int] | None=None) -> Iterable[tuple[int]]:
  if special_tokens is not None and len(special_tokens) > 0:
    pattern = '|'.join(re.escape(token) for token in special_tokens)
    if special_token_to_index is not None:
      pattern = "(" + pattern + ")"
    for chunk in re.split(pattern, text):
      if special_token_to_index is not None and chunk in special_token_to_index:
        yield (special_token_to_index[chunk],)
        continue
      for matched_pre_token in re.finditer(PAT, chunk):
        yield tuple(byte for byte in matched_pre_token.group(0).encode('utf-8'))
  else:
    for matched_pre_token in re.finditer(PAT, text):
      yield tuple(byte for byte in matched_pre_token.group(0).encode('utf-8'))


def merge_frq_tbls(frq_tbls: list[dict[tuple[int], int]]) -> list[list[list[int], int]]:
  union = frq_tbls[0]
  for counter in frq_tbls[1:]:
    union.update(counter)
  
  ret = []
  for bytes_in_word, cnt in union.items():
    ret.append([list(bytes_in_word), cnt])
  return ret


def pre_tokenize_and_count(text: str, special_tokens=["<|endoftext|>"]) -> list[list[list[int], int]]:
  ret = []
  cnts = Counter(pre_tokenize(text, special_tokens))
  for bytes_in_word, cnt in cnts.items():
    ret.append([list(bytes_in_word), cnt])
  return ret

def pre_tokenize_counter(text: str, special_tokens=["<|endoftext|>"]) -> list[list[list[int], int]]:
   return Counter(pre_tokenize(text, special_tokens))

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
      return pre_tokenize_and_count(txt, special_tokens)
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

  frq_tbl = create_frq_tbl_from_file(input_path, special_tokens, num_processes) if input_path is not None else pre_tokenize_and_count(debug_text, special_tokens)
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

class Tokenizer:
  def _add_missing_special_tokens_to_vocab(self, special_tokens: list[str] | None):
    if special_tokens is None:
        return
    assert type(special_tokens) == list
    for token in special_tokens:
      found = False
      for idx, bs in self.vocab.items():
        if bs.decode("utf-8", "ignore") == token:
          self.special_token_to_index[token] = idx
          found = True
          break
      if not found:
        self.special_token_to_index[token] = len(self.vocab)
        self.vocab[len(self.vocab)] = token.encode("utf-8")

        
  def __init__(self, vocab: dict[int, bytes], merges:list[tuple[bytes, bytes]], special_tokens: list[str] | None  = None):
    """Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
    """
    self.vocab = vocab
    self.special_tokens = special_tokens
    self.special_token_to_index = dict()
    self._add_missing_special_tokens_to_vocab(special_tokens)
    self.bytes_to_index = {bts: idx for idx, bts in self.vocab.items()}
    self.merges = [(self.bytes_to_index[bts1], self.bytes_to_index[bts2], self.bytes_to_index[bts1 + bts2]) for bts1, bts2 in merges]

  @classmethod
  def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens: list[str] | None = None) -> "Tokenizer":
    """Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
      (in the same format that your BPE training code output) and (optionally) a list of special tokens.
    """
    with open(vocab_filepath, "rb") as f:
      vocab = pickle.load(f)
    assert type(vocab) == dict
    with open(merges_filepath, "rb") as f:
        merges = pickle.load(f)
    assert type(merges) == list
    return cls(vocab, merges, special_tokens)
    
  
  def _merge(self, merge_pair: tuple[int, int], new_idx:int, indexes: list[int]) -> list[int]:
    if len(indexes) < 2:
       return indexes
    new_word = []
    i = 0
    while i < len(indexes):
      if i + 1 < len(indexes) and indexes[i] == merge_pair[0] and indexes[i+1] == merge_pair[1]:
        new_word.append(new_idx)
        i += 2
      else:
        new_word.append(indexes[i])
        i += 1
    return new_word
            

  def _encode_word(self, word: list[bytes]) -> list[int]:
    indexes = [self.bytes_to_index[bytes([byte])] for byte in word]
    for merge_pair_1, merge_pair_2, new_idx in self.merges:
      indexes = self._merge((merge_pair_1, merge_pair_2), new_idx, indexes)
    return indexes

  def _encode(self, text: str) -> Iterable[int]:
    for word in pre_tokenize(text, self.special_tokens, self.special_token_to_index):
        if len(word) == 1 and word[0] in self.special_token_to_index.values():
           yield word[0]
           continue
        for idx in self._encode_word(word):
          yield idx

  def encode(self, text: str) -> list[int]:
    """Encode an input text into a sequence of token IDs."""
    return list(self._encode(text))
  
  def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
    """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
    This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory."""
    for chunk in iterable:
      for idx in self._encode(chunk):
        yield idx

    
  
  def decode(self, ids: list[int]) -> str:
    """Decode a sequence of token IDs into text."""
    bts = [self.vocab[idx] for idx in ids]
    return b"".join(bts).decode("utf-8", errors='replace')


    


    


    