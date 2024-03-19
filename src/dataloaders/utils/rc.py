"""Utility functions for reverse complementing DNA sequences.

"""

from random import random

STRING_COMPLEMENT_MAP = {
    "A": "T", "C": "G", "G": "C", "T": "A", "a": "t", "c": "g", "g": "c", "t": "a",
    "N": "N", "n": "n",
}

def coin_flip(p=0.5):
    """Flip a (potentially weighted) coin."""
    return random() > p


def string_reverse_complement(seq):
    """Reverse complement a DNA sequence."""
    '''
    这个函数接收一个 DNA 序列字符串 seq 作为输入，并返回该序列的反转互补序列。
    它首先初始化一个空字符串 rev_comp 用于存储反转互补序列。
    然后，它遍历输入序列的每个碱基，但是是反向的（seq[::-1]）。
    对于每个碱基，如果它在 STRING_COMPLEMENT_MAP 中有对应的互补碱基，则将其互补碱基添加到 rev_comp 字符串中。
    如果输入序列中包含非标准碱基（不在互补映射字典中），则保留原始碱基。
    最后，返回构建好的反转互补序列。
    '''
    rev_comp = ""
    for base in seq[::-1]:
        if base in STRING_COMPLEMENT_MAP:
            rev_comp += STRING_COMPLEMENT_MAP[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp
