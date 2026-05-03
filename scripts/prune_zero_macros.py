import csv

IN_FILE = 'ingridients_macro_all.csv'
OUT_FILE = 'ingridients_macro_pruned.csv'

def is_all_zero(values):
    for v in values:
        try:
            if float(v) != 0.0:
                return False
        except Exception:
            # non-numeric treated as non-zero to be safe
            return False
    return True

def main():
    kept = 0
    removed = 0
    with open(IN_FILE, encoding='utf-8', newline='') as inf, open(OUT_FILE, 'w', encoding='utf-8', newline='') as outf:
        reader = csv.reader(inf)
        writer = csv.writer(outf)
        header = next(reader, None)
        if header:
            writer.writerow(header)
        for row in reader:
            if not row:
                continue
            name = row[0]
            vals = row[1:5]
            if is_all_zero(vals):
                removed += 1
                continue
            writer.writerow([name] + vals)
            kept += 1
    print(f'Wrote {kept} rows to {OUT_FILE}, removed {removed} zero-stub rows')

if __name__ == '__main__':
    main()
