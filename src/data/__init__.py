"""Data layer for gamma capture fund."""
from .types import DataWithSource
from .download import download_spx, download_vix, load_data

__all__ = ["DataWithSource", "download_spx", "download_vix", "load_data"]
