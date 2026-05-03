import csv
import re
import difflib
from collections import defaultdict

MACRO_CSV = 'ingridients_macro.csv'
MISSING_TXT = 'missing_ingredients.txt'
OUT_FILLED = 'ingridients_macro_filled.csv'
OUT_APPENDED = 'ingridients_macro_appended.csv'
REMAINING = 'remaining_missing.txt'

DESCRIPTORS = [
    'sliced','diced','chopped','minced','fresh','frozen','canned','dried','crushed','ground',
    'grated','peeled','skinless','boneless','low-fat','reduced-fat','unsalted','salted','light',
    'extra-virgin','shredded','thinly','thick','small','large','medium','roasted','toasted',
    'raw','cooked','halved','quartered','whole','sweetened','unsweetened','powdered'
]

def normalize(name):
    if not name:
        return ''
    n = name.lower()
    n = re.sub(r"\(.*?\)", "", n)
    n = re.sub(r"[^a-z0-9\s'-]", ' ', n)
    # remove descriptors
    for d in DESCRIPTORS:
        n = re.sub(r'\b' + re.escape(d) + r'\b', ' ', n)
    # remove simple measurements/units and numbers
    n = re.sub(r'\b(cups?|cup|tbsp|tablespoon|tsp|teaspoon|grams|gram|g|kg|ml|l|oz|ounces?)\b', ' ', n)
    n = re.sub(r'\b\d+[\d\./-]*\b', ' ', n)
    n = re.sub(r'\s+', ' ', n).strip()
    return n

def load_macros(path):
    macros = {}
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row: continue
            name = row[0].lower().strip()
            macros[name] = row[1:]
    return macros

def try_match(name, macros, macro_keys):
    # exact
    if name in macros:
        return name
    # singular heuristic
    if name.endswith('es') and name[:-2] in macros:
        return name[:-2]
    if name.endswith('s') and name[:-1] in macros:
        return name[:-1]
    # token-based: try each token
    tokens = name.split()
    for t in tokens[::-1]:
        if t in macros:
            return t
        if t.endswith('s') and t[:-1] in macros:
            return t[:-1]
    # fuzzy match
    close = difflib.get_close_matches(name, macro_keys, n=1, cutoff=0.8)
    if close:
        return close[0]
    # try shorter substrings
    for length in range(len(tokens), 0, -1):
        for i in range(len(tokens)-length+1):
            sub = ' '.join(tokens[i:i+length])
            if sub in macros:
                return sub
    return None

def main():
    macros = load_macros(MACRO_CSV)
    macro_keys = list(macros.keys())
    missing = []
    with open(MISSING_TXT, encoding='utf-8') as f:
        for line in f:
            n = line.strip()
            if n:
                missing.append(n)

    filled = {}
    remaining = []
    for name in missing:
        n = normalize(name)
        if not n:
            remaining.append(name)
            continue
        mapped = try_match(n, macros, macro_keys)
        if mapped:
            filled[name] = (mapped, macros[mapped])
        else:
            remaining.append(name)

    # write filled CSV (unique filled entries)
    with open(OUT_FILLED, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ingr_name','cal/g','fat(g)','carb(g)','protein(g)'])
        for original, (mapped, vals) in filled.items():
            writer.writerow([original] + vals)

    # create appended CSV: original macros + filled mapped (avoid duplicates)
    appended = dict(macros)
    for original, (mapped, vals) in filled.items():
        key = original.lower()
        if key not in appended:
            appended[key] = vals
    with open(OUT_APPENDED, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ingr_name','cal/g','fat(g)','carb(g)','protein(g)'])
        for name, vals in appended.items():
            writer.writerow([name] + vals)

    with open(REMAINING, 'w', encoding='utf-8') as f:
        for r in remaining:
            f.write(r + '\n')

    print(f'Total missing scanned: {len(missing)}')
    print(f'Filled by mapping: {len(filled)}')
    print(f'Remaining unfilled: {len(remaining)}')

if __name__ == '__main__':
    main()
