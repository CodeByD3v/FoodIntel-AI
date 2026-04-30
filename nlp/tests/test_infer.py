import yaml
import torch
import torch.nn.functional as F

import infer as infer_module
from infer import encode_ingredients


class DummyEncoder:
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs

    def load_state_dict(self, *args, **kwargs):
        return None

    def eval(self):
        return self

    def __call__(self, ingredient_str, ingredient_list):
        base = torch.arange(1, 513, dtype=torch.float32).unsqueeze(0)
        return F.normalize(base, p=2, dim=1)


def test_full_infer_with_dummy_random_init_weights(tmp_path, monkeypatch):
    config = {
        "model": {
            "bert_model_name": "bert-base-uncased",
            "freeze_bert": True,
            "output_dim": 512,
            "dropout": 0.1,
        },
        "usda": {
            "features_path": str(tmp_path / "missing_features.parquet"),
            "descriptions_path": str(tmp_path / "missing_descriptions.json"),
            "scaler_path": str(tmp_path / "missing_scaler.json"),
            "score_cutoff": 70,
            "cache_size": 2000,
            "nutrient_ids": [1008, 1003, 1004, 1005, 1079, 1093, 1253, 1258, 1087, 1089, 1162, 1106, 1109, 1110, 1180, 1175, 1178],
        },
        "ner": {"spacy_model": "en_core_web_sm", "fallback": "comma_split"},
    }
    config_path = tmp_path / "nlp.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(infer_module, "BERTIngredientEncoder", DummyEncoder)

    output = encode_ingredients(
        "chicken, basmati rice, olive oil",
        model_path=str(tmp_path / "dummy_missing.pth"),
        config_path=str(config_path),
    )

    expected_keys = {
        "raw_input",
        "cleaned_input",
        "extracted_ingredients",
        "usda_matches",
        "nutrition_context",
        "embedding",
        "embedding_norm",
    }
    assert expected_keys.issubset(output.keys())
    assert len(output["embedding"]) == 512
    assert abs(output["embedding_norm"] - 1.0) < 1e-5
    assert len(output["extracted_ingredients"]) >= 1
