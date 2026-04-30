from ner import IngredientNER


def test_ner_common_food_phrases():
    ner = IngredientNER()
    assert ner.extract("chicken, rice, olive oil") == ["chicken", "rice", "olive oil"]


def test_ner_single_ingredient():
    ner = IngredientNER()
    assert ner.extract("salmon") == ["salmon"]


def test_ner_fallback_comma_split():
    ner = IngredientNER()
    assert ner.extract("unknownalpha, unknownbeta") == ["unknownalpha", "unknownbeta"]


def test_normalize_ingredient_variant():
    ner = IngredientNER()
    assert ner.normalize_ingredient("evoo") == "olive oil"

