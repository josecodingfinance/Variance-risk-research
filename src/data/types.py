"""Data types for gamma capture fund."""

from dataclasses import dataclass
import pandas as pd


@dataclass
class DataWithSource:
    """
    DataFrame paired with a flag indicating if data is synthetic.

    This wrapper survives all pandas operations (slicing, loc, arithmetic, etc.)
    unlike df.attrs, which is lost in many operations. Ensures downstream code
    always knows whether it's using real data or synthetic fallback.
    """

    df: pd.DataFrame
    is_synthetic: bool
