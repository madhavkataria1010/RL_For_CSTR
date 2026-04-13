"""Download official paper-style figure assets used as visual references."""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path

from src.utils.paths import FIGURES_DIR


OFFICIAL_BASE = "https://raw.githubusercontent.com/OptiMaL-PSE-Lab/CIRL/c3fd22580a15d1e008570c08e78b38cdd887ef2c/plots"
REFERENCE_FILES = [
    "network_size_analysis.pdf",
    "lc_sp_newobs_0306.pdf",
]


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        target.write_bytes(response.read())


def _convert_pdf_to_png(pdf_path: Path) -> Path | None:
    if shutil.which("sips") is None:
        return None
    png_path = pdf_path.with_suffix(".png")
    if png_path.exists():
        return png_path
    subprocess.run(
        ["sips", "-s", "format", "png", str(pdf_path), "--out", str(png_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return png_path if png_path.exists() else None


def main() -> None:
    output_dir = FIGURES_DIR / "official"
    generated: list[Path] = []
    for filename in REFERENCE_FILES:
        pdf_path = output_dir / filename
        _download(f"{OFFICIAL_BASE}/{filename}", pdf_path)
        generated.append(pdf_path)
        png_path = _convert_pdf_to_png(pdf_path)
        if png_path is not None:
            generated.append(png_path)
    for path in generated:
        print(f"generated: {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
