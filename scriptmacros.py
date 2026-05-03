import pandas as pd
import re

INPUT_FILE = "ingridients_macro.csv"
OUTPUT_FILE = "final_macros.csv"

# =========================
# REMOVE NON-FOOD / JUNK
# =========================
REMOVE_TYPES = {
    "beverage", "drink"
}

# =========================
# CANONICAL MAP (IMPORTANT)
# =========================
CANONICAL = {
    "chicken breast": "chicken",
    "whole chicken": "chicken",
    "ground chicken": "chicken",

    "beef steak": "beef",
    "ground beef": "beef",

    "scrambled eggs": "egg",
    "boiled eggs": "egg",

    "cheddar cheese": "cheese",
    "mozzarella cheese": "cheese",

    "white rice": "rice",
    "brown rice": "rice",

    "spaghetti": "pasta",
    "macaroni": "pasta",

    "olive oil": "oil",
    "vegetable oil": "oil"
}

# =========================
# CLEAN NAME
# =========================
def clean_name(name):
    name = name.lower().strip()

    # remove special chars
    name = re.sub(r'[^a-z\s]', '', name)

    # remove noise words
    for word in ["fresh", "chopped", "raw", "cooked"]:
        name = name.replace(word, "")

    name = re.sub(r'\s+', ' ', name).strip()

    return CANONICAL.get(name, name)


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT_FILE)

# =========================
# REMOVE SYNTHETIC / JUNK
# =========================
df = df[df["is_synthetic"] == False]

# optional: remove drinks
df = df[~df["food_type"].isin(REMOVE_TYPES)]

# =========================
# CLEAN NAMES
# =========================
df["ingr_name"] = df["food_name"].apply(clean_name)

# =========================
# ASSUME SERVING = 100g (IMPORTANT)
# =========================
# If your dataset is already per 100g → this is correct
# If not → you must adjust

df["cal/g"] = df["calories"] / 100
df["fat(g)"] = df["fat_g"] / 100
df["carb(g)"] = df["carbs_g"] / 100
df["protein(g)"] = df["protein_g"] / 100

# =========================
# KEEP ONLY REQUIRED COLUMNS
# =========================
df = df[["ingr_name", "cal/g", "fat(g)", "carb(g)", "protein(g)"]]

# =========================
# REMOVE BAD ROWS
# =========================
df = df[df["ingr_name"].notnull()]
df = df[df["cal/g"] > 0]

# =========================
# MERGE DUPLICATES
# =========================
df = df.groupby("ingr_name").mean().reset_index()

# =========================
# SAVE
# =========================
df.to_csv(OUTPUT_FILE, index=False)

print("🔥 FINAL MACRO DATASET READY:", OUTPUT_FILE)