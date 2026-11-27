"""Reload the actual trained networks without executable model serialization."""

import numpy as np
import pytest
from test_model import SmallDataset, small_model

from financial_ts.bundles import NumericalPredictor, create_bundle, preprocess, verify_bundle
from financial_ts.model import _Windows


def metadata(**overrides):
    return {
        "source_revision": "fixture-source",
        "data_fingerprint": {"sha256": "test-data"},
        "target": "close[t+2]/close[t+1]-1",
        "universe": ["AAA", "BBB"],
        "fit_end": "2024-01-04T16:00:00+08:00",
        "seed": 23,
        "usage_rights": {"model": "research_only", "data": "synthetic"},
        **overrides,
    }


def neural_bundle(path):
    data = SmallDataset()
    model = small_model().fit(data)
    sampler = model._features(data, "test", "infer")
    windows = _Windows(sampler, 2)
    features = np.stack([windows[i][0].numpy() for i in range(len(windows))])
    padding = np.stack([windows[i][1].numpy() for i in range(len(windows))])
    predicted = model.predict(data).to_numpy()
    result = create_bundle(
        model,
        path,
        feature_names=["x", "y"],
        preprocessing={"kind": "identity"},
        metadata=metadata(),
        sequence_length=4,
        fixture_features=features,
        fixture_padding=padding,
        fixture_scores=predicted,
    )
    return result, features, padding, predicted


def test_numerical_network_round_trip_matches_original(tmp_path):
    result, features, padding, original = neural_bundle(tmp_path / "bundle")
    assert result["reload_parity"]["status"] == "passed"
    loaded = NumericalPredictor(tmp_path / "bundle")
    np.testing.assert_allclose(
        loaded.predict(features, padding=padding), original, atol=1e-7, rtol=0
    )
    assert not list((tmp_path / "bundle").glob("*.pkl"))
    with pytest.raises(ValueError, match="Feature order"):
        loaded.predict(features, padding=padding, feature_names=["y", "x"])
    with pytest.raises(ValueError, match="context length"):
        loaded.predict(features[:, :2], padding=padding[:, :2])
    assert verify_bundle(tmp_path / "bundle")["bundle_id"] == result["bundle_id"]


def test_tampered_and_unregistered_artifacts_rejected(tmp_path):
    neural_bundle(tmp_path / "bundle")
    extra = tmp_path / "bundle" / "unexpected.pkl"
    extra.write_bytes(b"not executable")
    with pytest.raises(ValueError, match="unregistered"):
        verify_bundle(tmp_path / "bundle")
    extra.unlink()
    (tmp_path / "bundle" / "metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        NumericalPredictor(tmp_path / "bundle")


def test_existing_version_is_never_overwritten(tmp_path):
    neural_bundle(tmp_path / "bundle")
    before = (tmp_path / "bundle" / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="immutable"):
        neural_bundle(tmp_path / "bundle")
    assert (tmp_path / "bundle" / "manifest.json").read_bytes() == before


def test_bad_reload_fixture_does_not_publish(tmp_path):
    model = small_model().fit(SmallDataset())
    with pytest.raises(ValueError, match="parity failed"):
        create_bundle(
            model,
            tmp_path / "bad",
            feature_names=["x", "y"],
            preprocessing={"kind": "identity"},
            metadata=metadata(),
            sequence_length=4,
            fixture_features=np.zeros((2, 4, 2)),
            fixture_scores=np.full(2, 1e10),
        )
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".bundle-*"))


def test_processor_all_missing_column_and_clip_matches_qlib():
    recipe = {
        "kind": "robust_zscore",
        "mean": [1.0, None, 2.0],
        "scale": [2.0, None, 0.5],
        "clip": 3.0,
        "fill_value": 0.0,
    }
    actual = preprocess(np.array([[np.inf, 5, 1], [100, np.nan, 2]]), recipe, 3)
    np.testing.assert_allclose(actual, [[0, 0, -2], [3, 0, 0]])


def test_symlinked_manifest_is_rejected(tmp_path):
    neural_bundle(tmp_path / "bundle")
    manifest = tmp_path / "bundle" / "manifest.json"
    other = tmp_path / "manifest.json"
    manifest.replace(other)
    manifest.symlink_to(other)
    with pytest.raises(ValueError, match="Symlink"):
        verify_bundle(tmp_path / "bundle")


def test_unknown_fit_cutoff_cannot_bypass_temporal_validation(tmp_path):
    model = small_model().fit(SmallDataset())
    with pytest.raises(ValueError, match="finite timestamp"):
        create_bundle(
            model,
            tmp_path / "bad",
            feature_names=["x", "y"],
            preprocessing={"kind": "identity"},
            metadata=metadata(fit_end="NaT"),
            sequence_length=4,
        )
