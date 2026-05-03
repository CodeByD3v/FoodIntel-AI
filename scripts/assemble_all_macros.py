import csv
import os

APPENDED = 'ingridients_macro_appended.csv'
ORIG = 'ingridients_macro.csv'
MISSING_STUBS = 'ingridients_macro_missing.csv'
REMAINING = 'remaining_missing.txt'
OUT = 'ingridients_macro_all.csv'

def read_csv_to_dict(path):
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row: continue
            key = row[0].lower().strip()
            d[key] = row[1:]
    return d

def main():
    # prefer appended file if present
    data = {}
    if os.path.exists(APPENDED):
        data = read_csv_to_dict(APPENDED)
        src = APPENDED
    else:
        data = read_csv_to_dict(ORIG)
        src = ORIG

    # add stubs from ingridients_macro_missing.csv if exists
    if os.path.exists(MISSING_STUBS):
        stubs = read_csv_to_dict(MISSING_STUBS)
        for k, v in stubs.items():
            if k not in data:
                data[k] = v

    # fallback: add remaining_missing.txt entries as zero-value stubs
    if os.path.exists(REMAINING):
        with open(REMAINING, encoding='utf-8') as f:
            for line in f:
                n = line.strip()
                if not n: continue
                k = n.lower().strip()
                if k not in data:
                    data[k] = ['0.00','0.00','0.00','0.00']

    # write combined file
    with open(OUT, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ingr_name','cal/g','fat(g)','carb(g)','protein(g)'])
        for name in sorted(data.keys()):
            writer.writerow([name] + data[name])

    print(f'Source used: {src}')
    print(f'Total ingredient macro rows written: {len(data)}')
    print(f'Output file: {OUT}')

if __name__ == '__main__':
    main()
