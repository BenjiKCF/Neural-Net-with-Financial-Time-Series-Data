"""Portfolio recording with an explicit cash benchmark when no ticker is given."""

from __future__ import annotations

import copy

import pandas as pd
from qlib.data import D
from qlib.workflow.record_temp import PortAnaRecord


class PortfolioRecord(PortAnaRecord):
    """Keep data-only YAML replayable while fixing Qlib's None benchmark fallback."""

    def __init__(self, recorder, config, **kwargs):
        config = copy.deepcopy(config)
        backtest = config["backtest"]
        if backtest.get("benchmark") is None:
            dates = pd.DatetimeIndex(
                D.calendar(
                    start_time=backtest["start_time"], end_time=backtest["end_time"], freq="day"
                )
            )
            backtest["benchmark"] = pd.Series(0.0, index=dates, name="cash")
        super().__init__(recorder=recorder, config=config, **kwargs)
