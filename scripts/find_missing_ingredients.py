import csv
import ast
import sys
from collections import Counter

macro_path = 'ingridients_macro.csv'
dataset_path = 'ultimate_dataset.csv'
missing_out = 'missing_ingredients.txt'
stubs_out = 'ingridients_macro_missing.csv'

def normalize(name):
    if not name:
        return ''
    return ' '.join(name.lower().strip().split())

def load_macros(path):
    names = set()
    try:
        with open(path, encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row: continue
                names.add(normalize(row[0]))
    except FileNotFoundError:
        print(f"Macro file not found: {path}")
        sys.exit(1)
    return names

def extract_ingredients(path):
    uniq = set()
    counter = Counter()
    with open(path, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        if 'ultimate_ingredients' not in reader.fieldnames:
            print('Expected column "ultimate_ingredients" not found in dataset headers:', reader.fieldnames)
            sys.exit(1)
        for i, row in enumerate(reader, 1):
            s = row.get('ultimate_ingredients')
            if not s: continue
            try:
                items = ast.literal_eval(s)
            except Exception:
                # fallback: try replace single quotes to double and parse as json-like
                try:
                    items = ast.literal_eval(s.replace('"', '"'))
                except Exception:
                    continue
            if not isinstance(items, (list, tuple)):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                name = it.get('ingredient') or it.get('ingr') or it.get('name')
                name = normalize(name)
                if name:
                    uniq.add(name)
                    counter[name] += 1
    return uniq, counter

def main():
    print('Loading macros...')
    macros = load_macros(macro_path)
    print(f'Found {len(macros)} macro entries')
    print('Scanning dataset for ingredients (this may take a while)...')
    uniq, counter = extract_ingredients(dataset_path)
    print(f'Found {len(uniq)} unique ingredient names in dataset')
    missing = sorted([n for n in uniq if n not in macros])
    print(f'{len(missing)} ingredients are missing from macros')
    with open(missing_out, 'w', encoding='utf-8') as f:
        for name in missing:
            f.write(name + '\n')
    # write stubs CSV with header
    with open(stubs_out, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ingr_name','cal/g','fat(g)','carb(g)','protein(g)'])
        for name in missing:
            writer.writerow([name,'0.00','0.00','0.00','0.00'])
    # print top 50 most common missing items
    print('\nTop 50 ingredients by frequency (dataset):')
    for name, cnt in counter.most_common(50):
        in_macro = 'YES' if name in macros else 'NO'
        print(f'{name}: {cnt} (in_macro={in_macro})')

if __name__ == '__main__':
    main()
