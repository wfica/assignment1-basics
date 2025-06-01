from collections.abc import Iterable
from collections import Counter
import regex as re
import heapq
from collections import defaultdict
from .pretokenization_example import find_chunk_boundaries
import multiprocessing

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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

  # Start merges
  while len(vocab) < vocab_size:
    # find the most common pair, first order by occurrences counts, resolve ties by lexicographic order
    most_frq_pair, _most_frq_pair_cnt = max(pairs_cnts.items(), key=lambda x: (x[1],pair_idx_to_pair_of_strings(x[0], vocab)))
    
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
            pairs_cnts[(old_indexes[i-1], new_token)] += word_cnt
            pairs_to_word_idx[(old_indexes[i-1], new_token)].append(word_idx)
          # update the following pair
          if i < len(old_indexes)-2:
            pairs_cnts[(old_indexes[i+1], old_indexes[i+2])] -= word_cnt
            pairs_cnts[(new_token, old_indexes[i+2])] += word_cnt
            pairs_to_word_idx[(new_token, old_indexes[i+2])].append(word_idx)
          i += 2
        else:
          new_indexes.append(old_token)
          i += 1
      frq_tbl[word_idx] = (new_indexes, word_cnt)

  return vocab, merges