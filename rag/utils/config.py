"""环境变量加载 + Tesseract OCR 配置。"""

from __future__ import annotations

import os


def load_dotenv(dotenv_path: str | None = None) -> None:
    """加载 .env 文件到 os.environ（无外部依赖，等效于 python-dotenv）。

    已存在的环境变量不会被覆盖。
    """
    if dotenv_path is None:
        dotenv_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        dotenv_path = os.path.abspath(dotenv_path)
    if not os.path.isfile(dotenv_path):
        return
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ[key] = value


def configure_tesseract() -> None:
    _TESSERACT_DIR = r"C:\Program Files\Tesseract-OCR"
    if os.path.isdir(_TESSERACT_DIR):
        if _TESSERACT_DIR not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _TESSERACT_DIR + os.pathsep + os.environ.get("PATH", "")
        # 优先用 Tesseract 安装目录自带的 tessdata（可通过 --list-langs 确认有 chi_sim）
        _default_tessdata_parent = _TESSERACT_DIR
    else:
        _default_tessdata_parent = None

    # 检查用户目录下是否有手动安装的语言包
    _user_tessdata = os.path.join(os.path.expanduser("~"), "tessdata")

    # 选择可用的 tessdata 路径：优先用户目录（有 chi_sim），否则用安装目录
    _tessdata_path = None
    if os.path.isfile(os.path.join(_user_tessdata, "chi_sim.traineddata")):
        _tessdata_path = _user_tessdata
    elif _default_tessdata_parent and os.path.isfile(
        os.path.join(_default_tessdata_parent, "tessdata", "chi_sim.traineddata")
    ):
        _tessdata_path = os.path.join(_default_tessdata_parent, "tessdata")
    elif _default_tessdata_parent:
        _tessdata_path = os.path.join(_default_tessdata_parent, "tessdata")

    if _tessdata_path:
        os.environ["TESSDATA_PREFIX"] = _tessdata_path

    # Tesseract 可执行文件路径兜底（unstructured_pytesseract 用 subprocess 调用）
    _TESSERACT_DIR = r"C:\Program Files\Tesseract-OCR"
    if os.path.isdir(_TESSERACT_DIR) and _TESSERACT_DIR not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _TESSERACT_DIR + os.pathsep + os.environ.get("PATH", "")
