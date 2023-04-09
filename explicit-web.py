import os
from typing import List

import datasets
from datasets.tasks import ImageClassification


logger = datasets.logging.get_logger(__name__)

_URL = ""

_HOMEPAGE = ""

_DESCRIPTION = ""

_CITATION = """"""


class CatsVsDogs(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("23.4.5")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "labels": datasets.features.ClassLabel(names=_NAMES),
                }
            ),
            supervised_keys=("image", "labels"),
            task_templates=[ImageClassification(image_column="image", label_column="labels")],
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        images_path = os.path.join(dl_manager.download_and_extract(_URL), "PetImages")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"files": dl_manager.iter_files([images_path])}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"files": dl_manager.iter_files([images_path])}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"files": dl_manager.iter_files([images_path])}
            ),
        ]

    def _generate_examples(self, files):
        for i, file in enumerate(files):
            if os.path.basename(file).endswith(".jpg"):
                with open(file, "rb") as f:
                    if b"JFIF" in f.peek(10):
                        yield str(i), {
                            "image": file,
                            "labels": os.path.basename(os.path.dirname(file)).lower(),
                        }
