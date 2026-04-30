from cleaner import IngredientCleaner


def test_cleaner_removes_quantities_and_units():
    cleaner = IngredientCleaner()
    assert cleaner.clean("2 cups chicken breast") == "chicken breast"


def test_cleaner_handles_fractions_and_commas():
    cleaner = IngredientCleaner()
    assert cleaner.clean("1/2 tsp salt, 3 tbsp olive oil") == "salt, olive oil"


def test_cleaner_handles_parentheses_and_decimals():
    cleaner = IngredientCleaner()
    assert cleaner.clean("garlic (minced), 1.5 oz butter") == "garlic, butter"


def test_cleaner_empty_string_no_crash():
    cleaner = IngredientCleaner()
    assert cleaner.clean("") == ""
    assert cleaner.split_ingredients("") == []

