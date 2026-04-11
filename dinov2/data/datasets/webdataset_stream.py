import json
import logging
import tarfile
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger("dinov2")


class WebDataset(IterableDataset):
    def __init__(
        self,
        *,
        root: str = "",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        shuffle: bool = False,
        image_ext: str = "png",
    ) -> None:
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform
        self.shuffle = shuffle
        self.image_ext = image_ext.lstrip(".").lower()
        self.shards = self._resolve_shards(Path(root))
        self.sample_count = self._count_samples(self.shards, self.image_ext)

        if self.sample_count == 0:
            raise ValueError(f'No samples with extension ".{self.image_ext}" found under "{root}"')

        logger.info(f'Indexed {self.sample_count} samples from {len(self.shards)} shard(s) under "{root}"')

    @staticmethod
    def _resolve_shards(root: Path) -> List[str]:
        if root.is_file():
            shard_paths = [root]
        else:
            shard_paths = sorted(root.rglob("*.tar"))

        if not shard_paths:
            raise ValueError(f'No shard files matching "*.tar" found under "{root}"')

        return [str(path) for path in shard_paths]

    @staticmethod
    def _count_samples(shards: List[str], image_ext: str) -> int:
        suffix = f".{image_ext}"
        sample_count = 0
        for shard in shards:
            with tarfile.open(shard, "r") as tar:
                for member in tar:
                    if member.isfile() and member.name.lower().endswith(suffix):
                        sample_count += 1
        return sample_count

    def _build_pipeline(self) -> Iterable[Tuple[object, torch.Tensor, str]]:
        try:
            import webdataset as wds
        except ImportError as exc:
            raise ImportError(
                "webdataset is required to use dataset_path=WebDataset:... "
                "Please install the `webdataset` package in this environment."
            ) from exc

        pipeline = wds.WebDataset(
            self.shards,
            shardshuffle=100 if self.shuffle else False,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
        )

        if self.shuffle:
            pipeline = pipeline.shuffle(1000)

        # Training does not use class labels, and our .cls payloads are strings
        # like "class0" that webdataset otherwise tries to parse as integers.
        pipeline = pipeline.map(self._drop_unused_fields)
        pipeline = pipeline.decode("pil")

        for sample in pipeline:
            image = sample.get(self.image_ext)
            if image is None:
                continue

            target = torch.zeros((1,))
            if self.transforms is not None:
                image, target = self.transforms(image, target)
            elif self.transform is not None:
                image = self.transform(image)
                if self.target_transform is not None:
                    target = self.target_transform(target)

            yield image, target, self._resolve_sample_name(sample)

    @staticmethod
    def _drop_unused_fields(sample):
        sample = dict(sample)
        sample.pop("cls", None)
        return sample

    def _resolve_sample_name(self, sample) -> str:
        metadata = sample.get("json")
        if isinstance(metadata, (bytes, bytearray)):
            try:
                metadata = json.loads(metadata.decode("utf-8"))
            except Exception:
                metadata = None

        if isinstance(metadata, dict):
            link_name = metadata.get("link_name")
            if isinstance(link_name, str) and link_name:
                return link_name

            source_path = metadata.get("source_path")
            if isinstance(source_path, str) and source_path:
                return source_path

        sample_key = sample.get("__key__")
        if isinstance(sample_key, str) and sample_key:
            return sample_key

        return "unknown_sample"

    def __iter__(self):
        return iter(self._build_pipeline())

    def __len__(self) -> int:
        return self.sample_count
