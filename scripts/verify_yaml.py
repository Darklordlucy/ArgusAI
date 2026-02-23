import yaml
from pathlib import Path

with open('Argus_AI/data/dataset.yaml') as f:
    d = yaml.safe_load(f)

train_path = Path(d['path']) / d['train']
val_path   = Path(d['path']) / d['val']

train_count = len(list(train_path.glob('*.jpg')) + list(train_path.glob('*.png')))
val_count   = len(list(val_path.glob('*.jpg'))   + list(val_path.glob('*.png')))

print('dataset.yaml verification:')
print(f'  Train path exists: {train_path.exists()}')
print(f'  Val path exists:   {val_path.exists()}')
print(f'  Train images:      {train_count}')
print(f'  Val images:        {val_count}')
print(f'  Classes:           {d["names"]}')

if train_count >= 1000:
    print('\n  READY FOR TRAINING')
elif train_count >= 500:
    print('\n  OK for POC training')
else:
    print('\n  WARNING: Low image count — download more datasets')