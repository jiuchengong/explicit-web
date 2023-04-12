import os
import datasets
import pandas as pd
from typing import List

from datasets.tasks import ImageClassification


logger = datasets.logging.get_logger(__name__)

_URL = ""

_HOMEPAGE = ""

_DESCRIPTION = ""

_CITATION = """"""

_NAMES = [
    'cartoon-intimacy', 'cartoon-neutral', 'cartoon-sexy', 'gamescreen',
    'feet', 'none', 'toys', 'male-sexy', 'neutral', 'female-sexy', 'porn',
    'dolls', 'male-erotic', 'cartoon-erotic', 'pulp', 'arts', 'erotic',
    'male-neutral-nudes', 'cum', 'child-nude', 'softcore',
    'cartoon-disgust', 'intimacy', 'texts', 'hentai', 'tatoo', 'pregnant',
    'bra', 'disgust', 'sm', 'comix', 'hentai-manga'
]


class CatsVsDogs(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("23.4.12")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name = 'small'),
        datasets.BuilderConfig(name = 'large'),
    ]
    DEFAULT_CONFIG_NAME = 'small'

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.features.ClassLabel(names=_NAMES),
                }
            ),
            supervised_keys=("image", "label"),
            task_templates=[ImageClassification(image_column="image", label_column="label")],
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        validset = dl_manager.download(f'{self.config.name}.valid.csv')
        trainset = dl_manager.download(f'{self.config.name}.train.csv')

        datadir = dl_manager.download('.')
        print(datadir, validset, trainset)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"meta": trainset, "datadir": datadir}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"meta": validset, "datadir": datadir}
            ),
        ]

    def _generate_examples(self, meta, datadir):
        dataset = pd.read_csv(meta)
        # dataset['image'] = datadir + '/' + dataset['image']
        for row in dataset.itertuples():
            yield str(row.Index), {
                "image": os.path.join(datadir, row.image),
                "label": row.label,
            }
