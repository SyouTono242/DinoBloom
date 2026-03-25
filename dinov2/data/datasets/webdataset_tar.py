import io
import json
import logging
import tarfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torchvision.datasets import VisionDataset

logger = logging.getLogger("dinov2")


class WebShardDataset(VisionDataset):
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
        super().__init__(root, transforms, transform, target_transform)
        self.image_ext = image_ext.lstrip(".").lower()
        self.samples = self._index_shards(Path(root), self.image_ext)
        self.true_len = len(self.samples)

        if self.true_len == 0:
            raise ValueError(f'No samples with extension ".{self.image_ext}" found under "{root}"')

        logger.info(f"Indexed {self.true_len} webdataset samples from {root}")

    @staticmethod
    def _index_shards(root: Path, image_ext: str) -> List[Dict[str, Optional[str]]]:
        if root.is_file():
            shard_paths = [root]
        else:
            shard_paths = sorted(root.rglob("*.tar"))
        if not shard_paths:
            raise ValueError(f'No shard files matching "*.tar" found under "{root}"')

        samples: List[Dict[str, Optional[str]]] = []
        expected_suffix = f".{image_ext}"

        for shard_path in shard_paths:
            logger.info(f"Indexing shard {shard_path}")
            grouped_members: Dict[str, Dict[str, Optional[str]]] = {}
            with tarfile.open(shard_path, "r") as tar:
                for member in tar:
                    if not member.isfile():
                        continue

                    member_path = Path(member.name)
                    suffix = member_path.suffix.lower()
                    if suffix not in {expected_suffix, ".cls", ".json"}:
                        continue

                    sample_key = str(member_path.with_suffix(""))
                    sample = grouped_members.setdefault(
                        sample_key,
                        {
                            "shard_path": str(shard_path),
                            "sample_key": sample_key,
                            "image_member": None,
                            "cls_member": None,
                            "json_member": None,
                        },
                    )

                    if suffix == expected_suffix:
                        sample["image_member"] = member.name
                    elif suffix == ".cls":
                        sample["cls_member"] = member.name
                    elif suffix == ".json":
                        sample["json_member"] = member.name

            for sample in grouped_members.values():
                if sample["image_member"] is not None:
                    samples.append(sample)

        return samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        try:
            image, source_name = self.get_image_data(index)
        except Exception as e:
            adjusted_index = index % self.true_len
            source_name = self.samples[adjusted_index]["sample_key"]
            print(f"can not read image for sample {index, e, source_name}")
            return self.__getitem__(index + 1)

        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, source_name

    def get_image_data(self, index: int) -> Tuple[Image.Image, str]:
        adjusted_index = index % self.true_len
        sample = self.samples[adjusted_index]
        shard_path = sample["shard_path"]
        image_member = sample["image_member"]

        if image_member is None:
            raise FileNotFoundError(f'Missing image member for sample "{sample["sample_key"]}"')

        with tarfile.open(shard_path, "r") as tar:
            extracted = tar.extractfile(image_member)
            if extracted is None:
                raise FileNotFoundError(f'Could not extract "{image_member}" from "{shard_path}"')

            image = Image.open(io.BytesIO(extracted.read())).convert(mode="RGB")

        return image, self._resolve_source_name(sample)

    def _resolve_source_name(self, sample: Dict[str, Optional[str]]) -> str:
        json_member = sample["json_member"]
        if json_member is None:
            return f'{sample["shard_path"]}:{sample["sample_key"]}'

        try:
            with tarfile.open(sample["shard_path"], "r") as tar:
                extracted = tar.extractfile(json_member)
                if extracted is None:
                    return f'{sample["shard_path"]}:{sample["sample_key"]}'
                metadata = json.loads(extracted.read().decode("utf-8"))
        except Exception:
            return f'{sample["shard_path"]}:{sample["sample_key"]}'

        link_name = metadata.get("link_name")
        if isinstance(link_name, str) and link_name:
            return link_name

        source_path = metadata.get("source_path")
        if isinstance(source_path, str) and source_path:
            return source_path

        return f'{sample["shard_path"]}:{sample["sample_key"]}'

    def get_target(self, index: int) -> torch.Tensor:
        return torch.zeros((1,))

    def __len__(self) -> int:
        return self.true_len
