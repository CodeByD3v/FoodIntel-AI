import pandas as pd
import re
import ast
from fractions import Fraction
from collections import defaultdict
from rapidfuzz import process
from tqdm import tqdm
import os

INPUT_FILE = "recipes.csv"
OUTPUT_FILE = "ultimate_dataset.csv"
CHUNK_SIZE = 3000

# 🔥 Delete old file (important)
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

# =========================
# CONFIG
# =========================
REMOVE = {"mix","stuff","base","product","item","thing","deli","miracle","frozen"}

CANONICAL = {
    "ground beef":"beef","hamburger":"beef",
    "chicken breast":"chicken","whole chicken":"chicken",
    "whole milk":"milk","skim milk":"milk",
    "brown rice":"rice","white rice":"rice",
    "spaghetti":"pasta","macaroni":"pasta",
    "cheddar cheese":"cheese","mozzarella":"cheese","parmesan":"cheese",
    "margarine":"butter","oleo":"butter","shortening":"butter",
}

TYPE = {
    "spice":["salt","pepper","cinnamon","nutmeg","cloves","oregano","yeast"],
    "fat":["butter","oil"],
    "liquid":["water","milk","broth","juice"],
    "protein":["beef","chicken","egg","fish"],
    "carb":["flour","sugar","rice","pasta"]
}

LIMITS = {
    "spice":10,
    "fat":400,
    "liquid":800,
    "protein":1500,
    "carb":1500,
    "other":1200
}

density = {"butter":227,"flour":120,"sugar":200,"milk":240}
count_weight = {"egg":50,"potato":150,"onion":110,"banana":120,"apple":180,"tomato":120}
package_weights = {"can":300,"pkg":200,"package":200,"box":250,"carton":500}

ner_cache = {}

# =========================
# HELPERS
# =========================

def clean_text(t):
    t = t.lower()
    t = re.sub(r'\(.*?\)', '', t)
    t = re.sub(r'[^a-z\s]', '', t)
    return re.sub(r'\s+', ' ', t).strip()

def parse_fraction(q):
    try:
        if ' ' in q:
            w,f=q.split()
            return int(w)+float(Fraction(f))
        if '/' in q:
            return float(Fraction(q))
        return float(q)
    except:
        return None

def parse_quantity_unit(text):
    text=text.lower()

    m=re.search(r'\((\d+\s?\d*/?\d*)\s*oz',text)
    if m:
        return parse_fraction(m.group(1)),"oz"

    m=re.search(r'(\d+)\s*to\s*(\d+)',text)
    if m:
        return (float(m.group(1))+float(m.group(2)))/2,"count"

    m=re.search(r'(\d+\s\d+/\d+|\d+/\d+|\d+)',text)
    if not m:
        return None,None

    qty=parse_fraction(m.group(1))

    if "cup" in text: return qty,"cup"
    if "tbsp" in text: return qty,"tbsp"
    if "tsp" in text: return qty,"tsp"
    if "oz" in text: return qty,"oz"
    if "lb" in text: return qty,"lb"

    for p in package_weights:
        if p in text:
            return qty,p

    return qty,"count"

def to_grams(qty,unit,ingredient):
    if qty is None:
        return None

    if unit=="oz": return qty*28
    if unit=="lb": return qty*454
    if unit=="cup": return qty*density.get(ingredient,240)
    if unit=="tbsp": return qty*15
    if unit=="tsp": return qty*5
    if unit in package_weights: return qty*package_weights[unit]

    if unit=="count":
        return qty*count_weight.get(ingredient,100)

    return None

def match_to_ner(raw, ner_list):
    if raw in ner_cache:
        return ner_cache[raw]
    m=process.extractOne(raw, ner_list)
    res=m[0] if m else None
    ner_cache[raw]=res
    return res

def clean_name(name):
    name=name.lower().strip()
    name=re.sub(r'[^a-z\s]','',name)

    if name in REMOVE or len(name)<2:
        return None

    return CANONICAL.get(name,name)

def get_type(name):
    for t,keys in TYPE.items():
        for k in keys:
            if k in name:
                return t
    return "other"

def fix_grams(name,grams):
    if grams is None or grams<=0:
        return None

    if name=="water":
        return None

    if name=="broth":
        grams*=0.3

    t=get_type(name)

    grams=min(grams,LIMITS.get(t,1200))

    return round(grams,2)

def merge(d):
    out=defaultdict(float)
    for k,v in d.items():
        out[k]+=v
    return [{"ingredient":k,"grams":round(v,2)} for k,v in out.items()]

# =========================
# MAIN
# =========================

print("🚀 Running FULL pipeline...")

first=True

for chunk in pd.read_csv(INPUT_FILE,chunksize=CHUNK_SIZE):

    rows=[]

    for r in tqdm(chunk.itertuples(),total=len(chunk)):

        try:
            dish=str(r.title)

            raw_list=ast.literal_eval(r.ingredients)
            ner_list=[x.lower().strip() for x in ast.literal_eval(r.NER)]

            merged=defaultdict(float)

            for raw in raw_list:
                raw_clean=clean_text(raw)

                qty,unit=parse_quantity_unit(raw)
                ing=match_to_ner(raw_clean,ner_list)

                if not ing:
                    continue

                ing=clean_name(ing)
                if not ing:
                    continue

                grams=to_grams(qty,unit,ing)
                grams=fix_grams(ing,grams)

                if not grams:
                    continue

                merged[ing]+=grams

            # ✅ FIX: allow even 1 ingredient (avoid empty dataset)
            if len(merged)==0:
                continue

            rows.append([dish,str(merge(merged))])

        except Exception:
            continue

    pd.DataFrame(rows,columns=["dish","ultimate_ingredients"]).to_csv(
        OUTPUT_FILE,
        mode="w" if first else "a",
        header=first,
        index=False
    )

    first=False

print("🔥 DONE → ultimate_dataset.csv ready")