#!/usr/bin/env python

import pandas as pd
from pathlib import Path
from jsonargparse import CLI
from tqdm import tqdm


def main(scheme:str='large.scheme', frac: float = 0.8, replace: bool = True):
    scheme = Path(scheme)

    classes = [line.split() for line in scheme.read_text().splitlines()]
    scheme.with_suffix('.class').write_text('\n'.join(l[1] for l in classes))

    classes = {l[0]:l[1] for l in classes if l[1] not in ['none']}

    sfxes = ['.jpeg', '.jpg', '.jpe', '.webp', '.web', '.png']
    images = [p for p in tqdm(Path('data').rglob('*.*')) if p.is_file() and p.suffix.lower() in sfxes]
    labels = [classes[p.parts[1]] for p in images]

    dataset = pd.DataFrame(dict(image=images, label=labels)).astype(str)
    dataset.to_csv(f'{scheme.stem}.csv', index=False, header=False)

    value_counts = dataset.label.value_counts()
    sz1, sz2 = value_counts.max(), value_counts.min()
    trainsize = int(sz1*frac) if replace else int(sz2*frac)
    print(value_counts)

    trainset = dataset.groupby('label').sample(n=trainsize, replace=replace)
    trainset.to_csv(f'{scheme.stem}.train.csv', index=False, header=False)

    validset = dataset.drop(trainset.index)
    trainset.to_csv(f'{scheme.stem}.valid.csv', index=False, header=False)
    print(trainset.shape, validset.shape)


if __name__ == "__main__":
    CLI(main)
