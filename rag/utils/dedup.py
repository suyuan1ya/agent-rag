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


def deduplicate(chunks: list[dict], threshold: int = 3) -> list[dict]:
    """对检索结果列表进行 SimHash 去重。

    Args:
        chunks: [{"text": ..., "page": ..., ...}, ...] 格式的结果列表
        threshold: 汉明距离阈值，默认 3（越小越严格）

    Returns:
        去重后的结果列表（保持原始顺序）
    """
    if not chunks:
        return []
    seen_fingerprints: list[int] = []
    result: list[dict] = []
    for chunk in chunks:
        fp = _simhash(chunk["text"])
        if any(_hamming(fp, existing) <= threshold for existing in seen_fingerprints):
            continue
        seen_fingerprints.append(fp)
        result.append(chunk)
    return result
