from collections import Counter
from typing import Any, Dict, List, Set, Tuple
import math

import regex


class BPETokenizer:
    def __init__(self, re_pattern: str):
        self.token_to_id: Dict[bytes, int] = {}
        self.id_to_token: Dict[int, bytes] = {}
        self.re_pattern = regex.compile(re_pattern)

    def _pretokenize(self, text: str) -> List[str]:
        words = self.re_pattern.findall(text)
        return words

    def train(self, corpus: str, vocab_size: int):
        words = self._pretokenize(corpus)
        # self._init_vocab(words)
        self._init_vocab(vocab_size)
        # 将每个单词都分割为字符：
        word_to_splits = self._split_words(words)
        while len(self.token_to_id) < vocab_size:
            # 统计所有邻接字符对及其频率：
            counter = self._count_pair_freqs(word_to_splits)
            # 找到频率最高的字符对并合并它们：
            most_common = counter.most_common(1)
            if not most_common:
                break
            self._merge_most_common_pair_for_training(most_common, word_to_splits)
        # 将字符对映射到整数ID以方便解码：
        for token, id_ in self.token_to_id.items():
            self.id_to_token[id_] = token

    def _init_vocab(self, vocab_size: int):
        if vocab_size < 2 ** 8:
            raise ValueError("Vocabulary size must be at least 256")

        for i in range(2 ** 8):
            self.token_to_id[bytes([i])] = i
            
    # def _init_vocab(self, words: List[str]):
    #     vocab = set()
    #     for word in words:
    #         for char in word.encode("utf-8"):
    #             byte = bytes([char])
    #             vocab.add(byte)
    #     for i, v in enumerate(vocab):
    #         self.token_to_id[v] = i

    def _split_words(self, words: List[str]) -> Dict[str, List[str]]:
        """  
        Returns:
            A dictionary mapping each word to its list of characters

            for example:
            {
                "hello": [b"h", b"e", b"l", b"l", b"o"],
                "world": [b"w", b"o", b"r", b"l", b"d"]
            }
        """
        word_to_splits: Dict[str, List[str]] = {}
        for word in words:
            split = self._split_word(word)
            word_to_splits[word] = split
        return word_to_splits

    def _split_word(self, word: str) -> List[str]:
        # split = [bytes([w]) for w in word.encode("utf-8")]
        split = []
        for w in word.encode("utf-8"):  # w 是一个整数，所以需要将其转换为字节序列
            split.append(bytes([w]))
        return split
    
    def _count_pair_freqs(self, word_to_splits: Dict[str, List[str]]) -> Counter:
        counter = Counter()
        for _, split in word_to_splits.items():
            for pair in zip(split[:-1], split[1:]):
                # counter[b"".join(pair)] += 1
                counter[pair] += 1
        return counter
    
    def _merge_most_common_pair_for_training(self, most_common: List[Tuple[Any]], word_to_splits: Dict[str, List[bytes]]):
        most_common_pair: Tuple[bytes] = most_common[0][0]
        self.token_to_id[b"".join(most_common_pair)] = len(self.token_to_id)
        for word, splits in word_to_splits.items():
            splits = self._merge_most_common_pair(most_common_pair, splits)
            word_to_splits[word] = splits

    def _merge_most_common_pair(self, most_common_pair: Tuple[bytes], splits: List[bytes]) -> List[bytes]:
        new_splits: List[bytes] = []
        i = 0
        while i < len(splits) - 1:
            if most_common_pair[0] == splits[i] and most_common_pair[1] == splits[i + 1]:
                new_splits.append(b"".join(most_common_pair))
                i += 2
            else:
                new_splits.append(splits[i])
                i += 1
        if i < len(splits):
            new_splits.append(splits[i])
        return new_splits

    def encode(self, inputs: str) -> List[List[str]]:
        words = self._pretokenize(inputs)
        word_to_splits = self._split_words(words)
        self._merge_most_common_pair_for_encoding(word_to_splits)
        outputs = []
        for _, splits in word_to_splits.items():
            output = [self.token_to_id[token] for token in splits]
            outputs.extend(output)
        return outputs
    
    def _merge_most_common_pair_for_encoding(self, word_to_splits: Dict[str, List[bytes]]) -> bool:
        for word, splits in word_to_splits.items():
            while True:
                most_common_pair = self._find_most_common_pair(splits)
                if not most_common_pair:
                    break
                splits = self._merge_most_common_pair(most_common_pair, splits)
            word_to_splits[word] = splits

    def _find_most_common_pair(self, splits: List[bytes]) -> Tuple[Tuple[bytes], float]:
        most_common_pair = None
        min_rank = float("inf")
        for pair in zip(splits[:-1], splits[1:]):
            rank = self.token_to_id.get(b"".join(pair), float("inf"))
            if rank < min_rank:
                min_rank = rank
                most_common_pair = pair
        return most_common_pair

    def decode(self, token_ids: List[int]) -> str:
        outputs = [self.id_to_token[id_].decode('utf-8') for id_ in token_ids]
        return "".join(outputs)