from pathlib import Path

print('=' * 50)
print('ARGUS-AI DATASET — END OF DAY 2 SUMMARY')
print('=' * 50)

for split in ['train', 'val']:
    base = Path(f'Argus_AI/data/{split}')
    imgs = list((base / 'images').glob('*.jpg')) + list((base / 'images').glob('*.png'))
    lbls = list((base / 'labels').glob('*.txt'))

    aug_suffixes = ['_rain', '_fog', '_night', '_blur']
    originals    = [i for i in imgs if not any(s in i.stem for s in aug_suffixes)]
    augmented    = [i for i in imgs if any(s in i.stem for s in aug_suffixes)]

    print(f'\n{split.upper()}:')
    print(f'  Total images:     {len(imgs)}')
    print(f'  Original images:  {len(originals)}')
    print(f'  Augmented copies: {len(augmented)}')
    print(f'  Label files:      {len(lbls)}')
    print(f'  Match:            {"OK" if len(imgs) == len(lbls) else "MISMATCH - run fix_errors.py"}')

counts = {0: 0, 1: 0, 2: 0}
names  = {0: 'pothole', 1: 'pedestrian', 2: 'obstacle'}

for lbl in Path('Argus_AI/data/train/labels').glob('*.txt'):
    for line in open(lbl):
        if line.strip():
            try:
                cid = int(line.split()[0])
                if cid in counts:
                    counts[cid] += 1
            except:
                continue

print('\nCLASS DISTRIBUTION (TRAIN):')
total = sum(counts.values())
for cid, count in counts.items():
    bar = '|' * int(count / total * 40) if total > 0 else ''
    print(f'  {names[cid]:12s}: {count:5d}  {bar}')

print('\n' + '=' * 50)