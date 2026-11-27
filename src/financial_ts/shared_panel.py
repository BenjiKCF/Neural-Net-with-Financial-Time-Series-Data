"""Memory-efficient Qlib windows for the feature-only neural training contract.

The shared sampler indexes one dataset-wide factor array instead of copying and
sorting all factor values for every segment. It retains Qlib's instrument-major
sample ordering, observed-date grid, membership holes, and NaN padding. Labels
continue through native Qlib's learning view and native label sampler.
"""

from __future__ import annotations

import bisect

import numpy as np
import pandas as pd
from qlib.data.dataset import TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP


class _FeatureStore:
    def __init__(self, frame: pd.DataFrame, dtype):
        if (
            not isinstance(frame.index, pd.MultiIndex)
            or frame.index.names != ["datetime", "instrument"]
            or not frame.index.is_unique
            or not isinstance(frame.columns, pd.MultiIndex)
        ):
            raise ValueError("Shared windows require unique datetime/instrument factor rows.")
        columns = np.flatnonzero(frame.columns.get_level_values(0) == "feature")
        if not len(columns):
            raise ValueError("The inference view has no feature columns.")
        values = frame.to_numpy(copy=False)
        if np.array_equal(columns, np.arange(columns[0], columns[-1] + 1)):
            # Avoid another feature-column copy when pandas exposes one array.
            values = values[:, columns[0] : columns[-1] + 1]
        else:
            values = values[:, columns]
        self.values = np.asarray(values, dtype=dtype)
        self.source_index = frame.index
        dates = frame.index.get_level_values("datetime")
        instruments = frame.index.get_level_values("instrument")
        self.dates = pd.DatetimeIndex(dates.unique()).sort_values()
        self.instruments = pd.Index(instruments.unique()).sort_values()
        self.date_codes = self.dates.get_indexer(dates)
        self.instrument_codes = self.instruments.get_indexer(instruments)
        self.grid = np.full((len(self.dates), len(self.instruments)), -1, dtype=np.int32)
        self.grid[self.date_codes, self.instrument_codes] = np.arange(len(frame), dtype=np.int32)
        # The native sampler sorts (instrument, datetime), then swaps index names.
        self.instrument_major_rows = np.lexsort((self.date_codes, self.instrument_codes))


class SharedPanelSampler:
    """Feature sampler whose windows reference one dataset-wide factor array."""

    def __init__(self, store: _FeatureStore, start, end, step_len: int):
        self.store = store
        self.step_len = step_len
        self.fillna_type = "none"
        rows = store.instrument_major_rows
        dates = store.source_index.get_level_values("datetime")
        keep = np.ones(len(rows), dtype=bool)
        if start is not None:
            keep &= dates[rows] >= pd.Timestamp(start)
        if end is not None:
            keep &= dates[rows] <= pd.Timestamp(end)
        self.rows = rows[keep]
        self.data_index = store.source_index[self.rows]

    def __len__(self):
        return len(self.rows)

    def get_index(self):
        return self.data_index

    def config(self, **kwargs):
        if set(kwargs) - {"fillna_type"} or kwargs.get("fillna_type", "none") != "none":
            raise ValueError("Shared neural windows support fillna_type='none' only.")
        self.fillna_type = "none"

    def _one(self, index):
        if isinstance(index, tuple):
            date, instrument = index
            row = bisect.bisect_right(self.store.dates, pd.Timestamp(date)) - 1
            column = self.store.instruments.get_loc(instrument)
        elif isinstance(index, (int, np.integer)) and 0 <= index < len(self.rows):
            source_row = self.rows[index]
            row = self.store.date_codes[source_row]
            column = self.store.instrument_codes[source_row]
        else:
            raise KeyError(f"Invalid shared-window index: {index!r}.")
        first = max(row - self.step_len + 1, 0)
        row_ids = self.store.grid[first : row + 1, column]
        padding = self.step_len - len(row_ids)
        result = np.full(
            (self.step_len, self.store.values.shape[1]), np.nan, dtype=self.store.values.dtype
        )
        valid = np.flatnonzero(row_ids >= 0)
        result[padding + valid] = self.store.values[row_ids[valid]]
        return result

    def __getitem__(self, index):
        if isinstance(index, (list, np.ndarray)):
            return np.stack([self._one(value) for value in index])
        return self._one(index)


class SharedTSDatasetH(TSDatasetH):
    """Opt-in dataset for models that read DK_I factors and DK_L labels only.

    In-memory learning factors are discarded because these models never request
    them. The on-disk handler cache is unchanged. Do not use this dataset for a
    tabular model that requires DK_L feature columns.
    """

    def setup_data(self, **kwargs):
        super().setup_data(**kwargs)
        # Reprocessing/replacing the handler must invalidate previous factor views.
        self._feature_stores = {}
        learn = getattr(self.handler, "_learn", None)
        if isinstance(learn, pd.DataFrame) and isinstance(learn.columns, pd.MultiIndex):
            columns = learn.columns.get_level_values(0)
            if "feature" in columns and "label" in columns:
                self.handler._learn = learn.loc[:, columns == "label"].copy()

    def _prepare_seg(self, slc, **kwargs):
        supported = {"col_set", "data_key", "dtype", "flt_col"}
        frame = getattr(self.handler, "_infer", None)
        use_shared = (
            kwargs.get("col_set") == "feature"
            and kwargs.get("data_key", DataHandlerLP.DK_I) == DataHandlerLP.DK_I
            and isinstance(frame, pd.DataFrame)
            and not getattr(self, "fetch_kwargs", None)
            and kwargs.get("flt_col", self.flt_col) is None
            and not set(kwargs) - supported
        )
        if not use_shared:
            return super()._prepare_seg(slc, **kwargs)
        if not isinstance(slc, slice):
            slc = slice(*slc)
        if slc.step is not None:
            return super()._prepare_seg(slc, **kwargs)
        # Preserve native pandas partial-string slicing outside the dated CN API.
        if any(isinstance(value, str) and len(value) != 10 for value in (slc.start, slc.stop)):
            return super()._prepare_seg(slc, **kwargs)
        dtype = np.dtype(kwargs.get("dtype") or frame.dtypes.iloc[0])
        stores = getattr(self, "_feature_stores", None)
        if stores is None:
            self._feature_stores = stores = {}
        key = dtype.str
        if key not in stores:
            stores[key] = _FeatureStore(frame, dtype)
        return SharedPanelSampler(stores[key], slc.start, slc.stop, self.step_len)
