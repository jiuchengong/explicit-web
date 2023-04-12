#!/usr/bin/env python

import pandas as pd
from pathlib import Path
from jsonargparse import CLI
from tqdm import tqdm


def main(scheme:str = 'large', frac: float = 0.8):
    replace = scheme == 'large'
    scheme = Path(scheme)

    classes = [line.split() for line in Path('scheme').read_text().splitlines()]
    classes = [line for line in classes if line[1] != 'none']

    scheme.with_suffix('.classes').write_text('\n'.join(list(set(c[1] for c in classes))))

    classes = {lc[0]:lc[1] for lc in classes}

    sfxes = ['.jpeg', '.jpg', '.jpe', '.webp', '.web', '.png']
    images = [p for p in tqdm(Path('data').rglob('*.*')) if p.is_file() and p.suffix.lower() in sfxes]
    labels = [classes.get(p.parts[1], 'none') for p in images]

    dataset = pd.DataFrame(dict(image=images, label=labels)).astype(str)
    dataset = dataset.loc[dataset.label!='none'].copy()

    dataset.to_csv(scheme.with_suffix('.full.csv'), index=False)

    value_counts = dataset.label.value_counts()
    sz1, sz2 = value_counts.max(), value_counts.min()
    trainsize = int(sz1*frac) if replace else int(sz2*frac)
    print(value_counts)

    trainset = dataset.groupby('label').sample(n=trainsize, replace=replace)
    trainset.to_csv(scheme.with_suffix('.train.csv'), index=False)

    validset = dataset.drop(trainset.index)
    validset.to_csv(scheme.with_suffix('.valid.csv'), index=False)
    print(trainset.shape, validset.shape)


if __name__ == "__main__":
    CLI(main)
