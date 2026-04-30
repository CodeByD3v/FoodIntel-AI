import json

import numpy as np
import pandas as pd

from usda_lookup import DEFAULT_NUTRIENT_IDS, USDALookup


def _write_lookup_artifacts(tmp_path):
    columns = [str(nutrient_id) for nutrient_id in DEFAULT_NUTRIENT_IDS]
    features = pd.DataFrame(
        [
            np.linspace(0.05, 0.95, len(columns)),
            np.linspace(0.01, 0.60, len(columns)),
        ],
        index=["100", "101"],
        columns=columns,
    )
    features_path = tmp_path / "usda_features.parquet"
    descriptions_path = tmp_path / "usda_descriptions.json"
    scaler_path = tmp_path / "usda_scaler.json"
    features.to_parquet(features_path)
    descriptions_path.write_text(
        json.dumps({"100": "Chicken breast raw", "101": "White rice cooked"}),
        encoding="utf-8",
    )
    scaler_path.write_text(json.dumps({column: {"min": 0.0, "max": 1.0} for column in columns}), encoding="utf-8")
    return features_path, descriptions_path, scaler_path


def test_lookup_returns_normalized_vector(tmp_path):
    paths = _write_lookup_artifacts(tmp_path)
    lookup = USDALookup(*map(str, paths), score_cutoff=50)
    vector = lookup.lookup("chicken breast")
    assert vector.shape == (17,)
    assert np.all(vector >= 0.0)
    assert np.all(vector <= 1.0)


def test_lookup_unknown_returns_zero_vector(tmp_path):
    paths = _write_lookup_artifacts(tmp_path)
    lookup = USDALookup(*map(str, paths), score_cutoff=95)
    vector = lookup.lookup("nonexistent_xyz_123")
    assert vector.shape == (17,)
    assert np.allclose(vector, np.zeros(17))


def test_lookup_second_call_served_from_cache(tmp_path):
    paths = _write_lookup_artifacts(tmp_path)
    lookup = USDALookup(*map(str, paths), score_cutoff=50)
    lookup.lookup("chicken breast")
    before = lookup.cache_info()
    lookup.lookup("chicken breast")
    after = lookup.cache_info()
    assert after.hits == before.hits + 1


def test_batch_lookup_shape(tmp_path):
    paths = _write_lookup_artifacts(tmp_path)
    lookup = USDALookup(*map(str, paths), score_cutoff=50)
    matrix = lookup.batch_lookup(["chicken", "rice"])
    assert matrix.shape == (2, 17)

