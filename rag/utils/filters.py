"""参考文献 / 实验数据过滤。"""

import re

# 参考文献段落头部关键词（短文本 + 关键词匹配）
_REF_HEADER_PATTERNS = [
    r'^参考文献\s*$', r'^References?\s*$', r'^REFERENCES\s*$',
    r'^Bibliography\s*$', r'^参考书目\s*$', r'^引用文献\s*$',
    r'^致\s*谢\s*$', r'^Acknowledgments?\s*$',
    r'^基金项目\s*$', r'^Funding\s*$',
    r'^作者贡献\s*$', r'^Author Contributions?\s*$',
    r'^版权声明\s*$', r'^Copyright\s*$',
    r'^补充材料\s*$', r'^Supplementary\s*$',
    r'^附录\s*$', r'^Appendix\s*$',
    r'^数据可[获取用]', r'^Data Availability\s*$',
    r'^利益冲突\s*$', r'^Competing Interests?\s*$',
]

# 正文章节头部关键词（碰到这些说明已离开参考文献区）
_SECTION_HEADER_PATTERNS = [
    r'^(引言|绪论|介绍|Introduction)\s*$',
    r'^(方法|实验|Method|Experiment|材料)\s*$',
    r'^(结果|Result|分析|Analysis|讨论|Discussion)\s*$',
    r'^(结论|总结|Conclusion|Summary)\s*$',
    r'^(摘要|Abstract|概要|Overview)\s*$',
    r'^(背景|Background|相关工作|Related Work|文献综述)\s*$',
    r'^[1-9]\.?\s+(引言|方法|结果|讨论|结论|介绍|背景|实验|总结)',
]


def _is_reference_header(text: str) -> bool:
    """检测是否进入参考文献 / 附录等非正文区域。"""
    stripped = text.strip()
    if len(stripped) > 80:
        return False
    for pat in _REF_HEADER_PATTERNS:
        if re.match(pat, stripped):
            return True
    return False


def _is_new_section_header(text: str) -> bool:
    """检测是否是新的正文章节标题（离开参考文献区）。"""
    stripped = text.strip()
    if len(stripped) > 80:
        return False
    for pat in _SECTION_HEADER_PATTERNS:
        if re.match(pat, stripped):
            return True
    return False


def _is_noise_block(text: str) -> bool:
    """检测是否是实验数据块、引用列表条目等无检索价值的噪声。

    规则：
    1. 参考文献条目：以 [数字] 开头 + 包含作者/期刊特征
    2. 纯数据块：数字+符号占比 > 55%
    """
    stripped = text.strip()
    tlen = len(stripped)
    if tlen < 30:
        return False

    # 规则 1：参考文献条目 — 以 [N] 或 [N,M] 或 [N-M] 开头
    if re.match(r'^\[\d+[,\-\d]*\]', stripped):
        return True

    # 规则 2：纯数据块 — 数字+空白+标点占比过高
    digit_sym = sum(1 for c in stripped if c.isdigit() or c in '±%.,;:<=>×()[] \t\n\r＋－±×÷＝≠≈＜＞≤≥±％‰℃℉')
    cjk = sum(1 for c in stripped if '一' <= c <= '鿿')
    alpha = sum(1 for c in stripped if c.isalpha() and c.isascii())

    # 极端情况：几乎全是数字符号
    if digit_sym / tlen > 0.70:
        return True

    # 数字符号过半 + 几乎没有中英文连续语义
    if digit_sym / tlen > 0.55 and (cjk + alpha) / tlen < 0.25:
        return True

    return False
