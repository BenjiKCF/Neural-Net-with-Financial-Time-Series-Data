"""Alpha158 features that can actually be computed from daily OHLCV."""

import numpy as np
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.loader import Alpha158DL
from qlib.data.dataset.processor import Processor


class FiniteFeatures(Processor):
    """Mark undefined factors missing without learning from any other row."""

    def __call__(self, df):
        df["feature"] = df["feature"].replace([np.inf, -np.inf], np.nan)
        return df


class OHLCVAlpha158(Alpha158):
    """157 of Alpha158's factors, omitting VWAP0 when true VWAP is unavailable.

    Daily OHLCV does not determine volume-weighted average execution price.
    Excluding that field avoids substituting a made-up value or an all-NaN column.
    """

    def get_feature_config(self):
        return Alpha158DL.get_feature_config(
            {
                "kbar": {},
                "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW"]},
                "rolling": {},
            }
        )
