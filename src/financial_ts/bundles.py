"""Immutable, numerical-only model bundles and reload parity.

Deployment never deserializes Qlib/MLflow pickle artifacts. Export receives a
trusted live model, then writes JSON, a LightGBM text model, or numeric NPZ state.
The supported preprocessing recipe is exported explicitly and fails closed if
new processors are added without an equivalent inference implementation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

BUNDLE_SCHEMA = 1


def canonical_json(value) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def file_hash(path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_bytes(canonical_json(value))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(canonical_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def environment_versions():
    packages = {}
    for name in ("financial-ts-qlib", "pyqlib", "numpy", "pandas", "torch", "lightgbm", "mlflow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def export_preprocessing(handler) -> tuple[list[str], dict]:
    """Encode exactly FiniteFeatures -> fitted RobustZScoreNorm -> Fillna."""
    processors = getattr(handler, "infer_processors", [])
    if [type(item).__name__ for item in processors] != [
        "FiniteFeatures",
        "RobustZScoreNorm",
        "Fillna",
    ]:
        raise ValueError("Bundle export requires the supported fitted inference processor recipe.")
    norm, fill = processors[1:]
    if norm.fields_group != "feature" or fill.fields_group != "feature":
        raise ValueError("Only feature-group preprocessing is supported.")
    columns = [str(item[-1] if isinstance(item, tuple) else item) for item in norm.cols]
    mean, scale = np.asarray(norm.mean_train), np.asarray(norm.std_train)
    if mean.shape != (len(columns),) or scale.shape != mean.shape:
        raise ValueError("Fitted processor shape differs from the feature schema.")

    # An all-missing training feature has NaN center/scale in Qlib and is filled
    # to zero after normalization. Preserve that behavior as explicit nulls.
    def encode(array):
        return [float(value) if np.isfinite(value) else None for value in array]

    return columns, {
        "kind": "robust_zscore",
        "mean": encode(mean),
        "scale": encode(scale),
        "clip": 3.0 if norm.clip_outlier else None,
        "fill_value": float(fill.fill_value),
        "fit_start": str(norm.fit_start_time),
        "fit_end": str(norm.fit_end_time),
        "nonfinite_input": "missing",
    }


def preprocess(features: np.ndarray, recipe: dict, count: int) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64).copy()
    if values.ndim not in (2, 3) or values.shape[-1] != count:
        raise ValueError("Features must have shape [stock, feature] or [stock, time, feature].")
    values[~np.isfinite(values)] = np.nan
    if recipe.get("kind") == "identity":
        fill_value = 0.0
    elif recipe.get("kind") == "robust_zscore":
        mean = np.array(recipe["mean"], dtype=np.float64)
        scale = np.array(recipe["scale"], dtype=np.float64)
        if mean.shape != (count,) or scale.shape != (count,):
            raise ValueError("Preprocessing dimensions differ from schema.")
        if (
            np.any(np.isfinite(scale) & (scale <= 0))
            or np.isinf(mean).any()
            or np.isinf(scale).any()
        ):
            raise ValueError("Invalid normalization statistics.")
        values = (values - mean) / scale
        if recipe.get("clip") is not None:
            clip = float(recipe["clip"])
            if not np.isfinite(clip) or clip <= 0:
                raise ValueError("Clipping bound must be positive and finite.")
            values = np.clip(values, -clip, clip)
        fill_value = float(recipe["fill_value"])
    else:
        raise ValueError("Unknown preprocessing recipe.")
    if not np.isfinite(fill_value):
        raise ValueError("Fill value must be finite.")
    return np.nan_to_num(values, nan=fill_value, posinf=fill_value, neginf=fill_value)


def _export_model(model, destination: Path) -> dict:
    name = type(model).__name__
    if name in {"SequenceForecastModel", "TemporalTransformerModel", "CrossSectionalForecastModel"}:
        if not model.fitted or model.model is None:
            raise ValueError("Cannot export an unfitted model.")
        if hasattr(model, "constructor_config"):
            config = model.constructor_config()
        else:
            fields = (
                "d_feat",
                "d_model",
                "nhead",
                "num_layers",
                "dropout",
                "architecture",
                "architecture_kwargs",
            )
            config = {key: getattr(model, key) for key in fields}
        config.update(device="cpu", num_threads=1)
        state = {
            key: tensor.detach().cpu().numpy() for key, tensor in model.model.state_dict().items()
        }
        if any(not np.isfinite(value).all() for value in state.values()):
            raise ValueError("Model contains nonfinite weights.")
        np.savez_compressed(destination / "weights.npz", **state)
        return {
            "kind": "cross_sectional" if name == "CrossSectionalForecastModel" else "sequence",
            "constructor": config,
            "target_mean": float(model.target_mean_),
            "target_scale": float(model.target_scale_),
        }
    if name == "LGBModel":
        if model.model is None:
            raise ValueError("Cannot export an unfitted LightGBM model.")
        model.model.save_model(str(destination / "booster.txt"))
        return {"kind": "lightgbm"}
    # Minimal numerical baseline contract; no sklearn pickle.
    if getattr(model, "kind", None) in {"zero", "ridge"}:
        if not getattr(model, "fitted_", getattr(model, "fitted", False)):
            raise ValueError("Cannot export an unfitted baseline.")
        coefficients = np.asarray(
            getattr(model, "coefficient_", getattr(model, "coef_", [])), dtype=np.float64
        )
        intercept = float(getattr(model, "intercept_", 0.0))
        if not np.isfinite(coefficients).all() or not np.isfinite(intercept):
            raise ValueError("Invalid baseline parameters.")
        return {"kind": model.kind, "coefficients": coefficients.tolist(), "intercept": intercept}
    raise ValueError(f"Unsupported or research-only model bundle: {name}")


def create_bundle(
    model,
    output,
    *,
    feature_names,
    preprocessing,
    metadata,
    sequence_length=1,
    fixture_features=None,
    fixture_padding=None,
    fixture_scores=None,
    tolerance=1e-6,
):
    """Export and independently reload an immutable candidate bundle.

    Metadata must declare source/data identity, label semantics, universe, fit
    cutoff, seed and usage rights. Unknown usage rights remain blocked at
    promotion; producing a research bundle never grants deployment rights.
    """
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError("Bundle destination already exists; versions are immutable.")
    names = list(feature_names)
    if (
        not names
        or len(set(names)) != len(names)
        or any(not isinstance(x, str) or not x for x in names)
    ):
        raise ValueError("Feature names must be nonempty, unique strings in training order.")
    if type(sequence_length) is not int or sequence_length < 1:
        raise ValueError("sequence_length must be positive.")
    required = {
        "source_revision",
        "data_fingerprint",
        "target",
        "universe",
        "fit_end",
        "seed",
        "usage_rights",
    }
    if not required.issubset(metadata):
        raise ValueError(f"Bundle metadata is missing {sorted(required - set(metadata))}.")
    if pd.isna(pd.Timestamp(metadata["fit_end"])):
        raise ValueError("Bundle fit_end must be a finite timestamp.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Reload tolerance must be positive and finite.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".bundle-", dir=output.parent))
    try:
        model_info = _export_model(model, temporary)
        write_json(temporary / "model.json", model_info)
        write_json(
            temporary / "schema.json", {"features": names, "sequence_length": sequence_length}
        )
        write_json(temporary / "preprocessing.json", preprocessing)
        write_json(temporary / "metadata.json", metadata)
        write_json(temporary / "environment.json", environment_versions())
        parity = {"status": "not_checked", "tolerance": tolerance}
        if fixture_features is not None:
            if fixture_scores is None:
                raise ValueError("Reload fixture requires independent original scores.")
            features = np.asarray(fixture_features, dtype=np.float64)
            expected = np.asarray(fixture_scores, dtype=np.float64).reshape(-1)
            padding = (
                np.zeros(features.shape[:-1], dtype=bool)
                if fixture_padding is None
                else np.asarray(fixture_padding, dtype=bool)
            )
            np.savez_compressed(
                temporary / "reload_fixture.npz",
                features=features,
                padding=padding,
                scores=expected,
            )
            # The temporary directory has only files created by this export.
            actual = NumericalPredictor(temporary, verify=False).predict(features, padding=padding)
            if actual.shape != expected.shape or not np.isfinite(expected).all():
                raise ValueError("Reload fixture scores are malformed.")
            error = float(np.max(np.abs(actual - expected))) if len(actual) else float("inf")
            if not np.isfinite(error) or error > tolerance:
                raise ValueError(f"Reload parity failed: max absolute error {error} > {tolerance}.")
            parity.update(status="passed", max_absolute_error=error, samples=len(expected))
        write_json(temporary / "parity.json", parity)
        files = {
            path.name: {"sha256": file_hash(path), "bytes": path.stat().st_size}
            for path in sorted(temporary.iterdir())
        }
        body = {"schema_version": BUNDLE_SCHEMA, "files": files}
        identity = hashlib.sha256(canonical_json(body)).hexdigest()
        write_json(temporary / "manifest.json", {**body, "bundle_id": identity})
        verify_bundle(temporary)
        os.rename(temporary, output)
        return {"bundle_id": identity, "path": str(output), "reload_parity": parity}
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_bundle(path) -> dict:
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Bundle must be a real directory.")
    manifest_path = path / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Symlinked bundle manifest is not allowed.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != BUNDLE_SCHEMA or not isinstance(
        manifest.get("files"), dict
    ):
        raise ValueError("Unsupported bundle manifest.")
    files = manifest["files"]
    required = {
        "model.json",
        "schema.json",
        "preprocessing.json",
        "metadata.json",
        "environment.json",
        "parity.json",
    }
    if not required.issubset(files):
        raise ValueError("Bundle is missing required artifacts.")
    if set(item.name for item in path.iterdir()) != set(files) | {"manifest.json"}:
        raise ValueError("Bundle contains missing or unregistered artifacts.")
    for name, identity in files.items():
        item = path / name
        if Path(name).name != name or item.is_symlink() or not item.is_file():
            raise ValueError("Unsafe artifact path in bundle.")
        if item.stat().st_size != identity["bytes"] or file_hash(item) != identity["sha256"]:
            raise ValueError(f"Bundle integrity check failed for {name}.")
    body = {"schema_version": manifest["schema_version"], "files": files}
    if hashlib.sha256(canonical_json(body)).hexdigest() != manifest.get("bundle_id"):
        raise ValueError("Bundle manifest identity does not match its contents.")
    return manifest


class NumericalPredictor:
    """Load only allowlisted architectures and numerical artifacts."""

    def __init__(self, bundle, *, verify=True):
        self.path = Path(bundle)
        self.manifest = verify_bundle(self.path) if verify else None
        self.schema = json.loads((self.path / "schema.json").read_text())
        self.recipe = json.loads((self.path / "preprocessing.json").read_text())
        self.info = json.loads((self.path / "model.json").read_text())
        self.metadata = json.loads((self.path / "metadata.json").read_text())
        kind = self.info["kind"]
        self.network = None
        if kind in {"sequence", "cross_sectional"}:
            import torch

            if kind == "sequence":
                from financial_ts.catalog import build_mini_encoder
                from financial_ts.model import SequenceForecastModel, _TemporalEncoder

                adapter = SequenceForecastModel(**self.info["constructor"])
                common = {
                    key: getattr(adapter, key)
                    for key in ("d_feat", "d_model", "nhead", "num_layers", "dropout")
                }
                network = (
                    _TemporalEncoder(**common)
                    if adapter.architecture == "transformer"
                    else build_mini_encoder(
                        adapter.architecture, **common, **adapter.architecture_kwargs
                    )
                )
            else:
                from financial_ts.cross_sectional import CrossSectionalForecastModel

                adapter = CrossSectionalForecastModel(**self.info["constructor"])
                network = adapter._build_network()
            with np.load(self.path / "weights.npz", allow_pickle=False) as archive:
                state = {key: torch.from_numpy(archive[key].copy()) for key in archive.files}
            network.load_state_dict(state, strict=True)
            self.network = network.cpu().eval()
            torch.set_num_threads(1)
        elif kind == "lightgbm":
            import lightgbm

            self.network = lightgbm.Booster(model_file=str(self.path / "booster.txt"))
        elif kind not in {"zero", "ridge"}:
            raise ValueError("Bundle model type is not allowlisted.")
        if self.manifest is not None and (self.path / "reload_fixture.npz").exists():
            parity = json.loads((self.path / "parity.json").read_text())
            with np.load(self.path / "reload_fixture.npz", allow_pickle=False) as archive:
                actual = self.predict(archive["features"], padding=archive["padding"])
                expected = archive["scores"]
            if actual.shape != expected.shape or not np.allclose(
                actual, expected, rtol=0, atol=parity["tolerance"]
            ):
                raise ValueError("Runtime reload parity failed under current code/dependencies.")

    def predict(self, features, *, padding=None, feature_names=None):
        if feature_names is not None and list(feature_names) != self.schema["features"]:
            raise ValueError("Feature order differs from the trained schema.")
        values = preprocess(features, self.recipe, len(self.schema["features"]))
        kind = self.info["kind"]
        if not len(values):
            raise ValueError("An inference batch must contain instruments.")
        if kind in {"sequence", "cross_sectional"}:
            import torch

            if values.ndim != 3 or values.shape[1] != self.schema["sequence_length"]:
                raise ValueError("Historical context length differs from trained sequence length.")
            missing = (
                np.zeros(values.shape[:2], dtype=bool)
                if padding is None
                else np.asarray(padding, dtype=bool)
            )
            if missing.shape != values.shape[:2] or missing[:, -1].any():
                raise ValueError("Padding shape is invalid or the decision-date row is absent.")
            values[missing] = 0
            from contextlib import nullcontext

            precision = self.info.get("constructor", {}).get("precision", "float32")
            autocast = (
                torch.autocast("cpu", dtype=torch.bfloat16)
                if precision == "bfloat16"
                else nullcontext()
            )
            with torch.inference_mode(), autocast:
                result = self.network(
                    torch.from_numpy(values.astype(np.float32)), torch.from_numpy(missing)
                )
            if isinstance(result, dict):
                result = result["score"]
            result = result.detach().float().cpu().numpy().reshape(-1)
            result = (
                result.astype(np.float64) * self.info["target_scale"] + self.info["target_mean"]
            )
        else:
            if values.ndim != 2:
                raise ValueError("Tabular model requires one feature row per instrument.")
            if kind == "zero":
                result = np.zeros(len(values))
            elif kind == "ridge":
                coefficients = np.asarray(self.info["coefficients"], dtype=float)
                if coefficients.shape != (values.shape[1],):
                    raise ValueError("Ridge coefficients differ from schema.")
                result = values @ coefficients + self.info["intercept"]
            else:
                result = self.network.predict(values)
        if np.asarray(result).shape != (len(values),) or not np.isfinite(result).all():
            raise ValueError("Inference produced nonfinite or incorrectly shaped scores.")
        return np.asarray(result, dtype=np.float64)


class InferenceBundleRecord:
    """Optional Qlib record: export a live trained model without loading a pickle."""

    def __init__(self, model, dataset, recorder, output_dir, metadata, tolerance=1e-6):
        self.model, self.dataset, self.recorder = model, dataset, recorder
        self.output = Path(output_dir)
        self.metadata, self.tolerance = metadata, tolerance

    def generate(self):
        import pandas as pd
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP
        from qlib.utils import class_casting

        handler = self.dataset.handler
        names, recipe = export_preprocessing(handler)
        # Reconstruct raw factor values from the source handler. A cached handler
        # may retain raw labels only: its processed features are still available,
        # so use an identity fixture over processed values for parity, while the
        # deployed bundle always retains the fitted normalization recipe.
        prediction = self.model.predict(self.dataset, segment="valid")
        date = pd.DatetimeIndex(prediction.index.get_level_values("datetime")).min()
        one_day = slice(date, date)
        sequence_length = getattr(self.dataset, "step_len", 1)
        kind = type(self.model).__name__
        if kind in {
            "SequenceForecastModel",
            "TemporalTransformerModel",
            "CrossSectionalForecastModel",
        }:
            from financial_ts.model import _Windows

            sampler = self.model._features(self.dataset, one_day, DataHandlerLP.DK_I)
            windows = _Windows(sampler, len(names))
            features = np.stack([windows[i][0].numpy() for i in range(len(windows))])
            padding = np.stack([windows[i][1].numpy() for i in range(len(windows))])
            index = sampler.get_index()
        else:
            with class_casting(self.dataset, DatasetH):
                frame = self.dataset.prepare(
                    one_day, col_set="feature", data_key=DataHandlerLP.DK_I
                )
            features, index = frame.to_numpy(dtype=float), frame.index
            padding = None
        # Export a separate processed-input parity receipt, then verify raw
        # preprocessing equivalence in the unit/integration acceptance tests.
        temporary = self.output.parent / ("." + self.output.name + "-parity")
        result = create_bundle(
            self.model,
            temporary,
            feature_names=names,
            preprocessing={"kind": "identity"},
            metadata=self.metadata,
            sequence_length=sequence_length,
            fixture_features=features,
            fixture_padding=padding,
            fixture_scores=prediction.reindex(index).to_numpy(),
            tolerance=self.tolerance,
        )
        # Recover input values whose normalization equals the retained processed
        # values, then validate with the real recipe before publishing the bundle.
        mean = np.asarray(recipe["mean"], dtype=float)
        scale = np.asarray(recipe["scale"], dtype=float)
        reconstructed = features * scale + mean
        try:
            result = create_bundle(
                self.model,
                self.output,
                feature_names=names,
                preprocessing=recipe,
                metadata={
                    **self.metadata,
                    "reload_fixture": "inverse-normalized processed validation features; source replay separately required",
                },
                sequence_length=sequence_length,
                fixture_features=reconstructed,
                fixture_padding=padding,
                fixture_scores=prediction.reindex(index).to_numpy(),
                tolerance=self.tolerance,
            )
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        self.recorder.save_objects(**{"inference_bundle.json": result})
        return result
