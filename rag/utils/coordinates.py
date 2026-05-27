from __future__ import annotations

"""从 Unstructured 元素中提取 PDF 坐标信息的辅助函数。"""


def _get_element_y_range(doc) -> tuple[float, float] | None:
    """从 Unstructured 元素中提取 y 坐标范围 (top_y, bottom_y)。"""
    try:
        meta = doc.metadata
        coords = meta.get('coordinates') or meta.get(' coordinates')
        if coords is None:
            return None
        points = coords.get('points')
        if not points:
            return None
        ys = [p[1] for p in points]
        return (min(ys), max(ys))
    except (KeyError, IndexError, TypeError):
        return None


def _get_page_height(doc) -> float | None:
    """从 Unstructured 元素元数据中提取页面高度。"""
    try:
        meta = doc.metadata
        coords = meta.get('coordinates') or meta.get(' coordinates')
        if coords is None:
            return None
        return coords.get('layout_height')
    except (KeyError, TypeError):
        return None
