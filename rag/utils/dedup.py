"""SimHash 文本去重（模块级工具函数）。"""

import hashlib


def _simhash(text: str, bits: int = 64) -> int:
    """计算文本的 SimHash 指纹，用于近似重复检测（替换 Jaccard）。"""
    if not text:
        return 0
    grams = [text[i:i + 3] for i in range(max(1, len(text) - 2))]
    v = [0] * bits
    for g in grams:
        h = int(hashlib.md5(g.encode('utf-8', errors='ignore')).hexdigest(), 16)
        for i in range(bits):
            if (h >> i) & 1:
                v[i] += 1
            else:
                v[i] -= 1
    fingerprint = 0
    for i in range(bits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


def _hamming(a: int, b: int) -> int:
    """计算两个 SimHash 指纹的汉明距离。"""
    return (a ^ b).bit_count()
