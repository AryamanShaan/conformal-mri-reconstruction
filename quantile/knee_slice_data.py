from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import List, NamedTuple

import h5py
import pytorch_lightning as pl
import torch

# from fastMRI.fastmri.data import transforms
from common import functions as transforms


class KneeSliceSample(NamedTuple):
    """
    One raw multicoil slice plus the target image and bookkeeping fields.
    """

    kspace: torch.Tensor
    target: torch.Tensor
    fname: str
    slice_num: int


class KneeSliceDataset(torch.utils.data.Dataset):
    """
    Minimal slice dataset for multicoil knee data.

    Each item is:
        - raw multicoil k-space cropped to (640, 320) - not done anymore
        - target RSS image cropped to (320, 320) - not done anymore
        - filename
        - slice index
    """

    def __init__(
        self,
        root: Path,
        # kspace_crop_shape: tuple[int, int] = (640, 640),
        # target_crop_shape: tuple[int, int] = (320, 320),
    ):
        self.root = root
        # self.kspace_crop_shape = kspace_crop_shape
        # self.target_crop_shape = target_crop_shape

        self.examples: List[tuple[Path, int]] = []
        h5_paths = sorted(root.glob("*.h5"))

        for h5_path in h5_paths:
            with h5py.File(h5_path, "r") as hf:
                num_slices = hf["kspace"].shape[0]
            self.examples.extend((h5_path, slice_idx) for slice_idx in range(num_slices))

    def __len__(self) -> int:
        return len(self.examples)

    # def _center_crop_or_pad_kspace(self, kspace_torch: torch.Tensor) -> torch.Tensor:
    #     """
    #     Make every multicoil slice match ``self.kspace_crop_shape``.

    #     If the slice is larger than the requested shape, we center-crop.
    #     If it is smaller, we place it in the center of a zero-filled tensor.
    #     This is done on the spatial dimensions only; the coil dimension and the
    #     final complex dimension are preserved.
    #     """
    #     assert (
    #         kspace_torch.ndim == 4
    #     ), (
    #         "Expected k-space tensor shape (num_coils, H, W, 2), hence 4 dimensions "
    #         f"are required; got {kspace_torch.ndim} dimensions."
    #     )
    #     assert (
    #         kspace_torch.shape[-1] == 2
    #     ), f"Expected final complex dimension to be 2, got {kspace_torch.shape[-1]}"

    #     target_h, target_w = self.kspace_crop_shape
    #     current_h, current_w = kspace_torch.shape[-3], kspace_torch.shape[-2]

    #     crop_h = min(current_h, target_h)
    #     crop_w = min(current_w, target_w)
    #     kspace_torch = transforms.complex_center_crop(kspace_torch, (crop_h, crop_w))

    #     if crop_h == target_h and crop_w == target_w:
    #         return kspace_torch

    #     output = torch.zeros(
    #         (kspace_torch.shape[0], target_h, target_w, kspace_torch.shape[-1]),
    #         dtype=kspace_torch.dtype,
    #     )

    #     pad_top = (target_h - crop_h) // 2
    #     pad_left = (target_w - crop_w) // 2
    #     pad_bottom = pad_top + crop_h
    #     pad_right = pad_left + crop_w

    #     output[:, pad_top:pad_bottom, pad_left:pad_right, :] = kspace_torch
    #     return output

    def __getitem__(self, index: int) -> KneeSliceSample:
        h5_path, slice_idx = self.examples[index]

        with h5py.File(h5_path, "r") as hf:
            kspace = hf["kspace"][slice_idx]
            target = hf["reconstruction_rss"][slice_idx]

        # Convert complex k-space to torch with real/imag in the last dimension,
        # then crop/pad every coil to a shared encoded size so batching works.
        kspace_torch = transforms.to_tensor(kspace)
        # kspace_torch = self._center_crop_or_pad_kspace(kspace_torch)

        # Target is already RSS image-space data. We still center-crop it to the
        # expected 320x320 to keep the training target shape fixed.
        target_torch = transforms.to_tensor(target)
        # target_torch = transforms.center_crop(target_torch, self.target_crop_shape)

        return KneeSliceSample(
            kspace=kspace_torch,
            target=target_torch,
            fname=h5_path.name,
            slice_num=slice_idx,
        )


class KneeSliceDataModule(pl.LightningDataModule):
    """
    Custom datamodule that batches raw multicoil slices.

    Masking is intentionally not done here. It happens inside the model forward
    so one mask can be applied to the full local batch.
    """

    def __init__(
        self,
        data_path: Path,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler

    def _make_dataset(self, split: str) -> KneeSliceDataset:
        return KneeSliceDataset(root=self.data_path / split)

    def _make_loader(
        self, split: str, shuffle: bool
    ) -> torch.utils.data.DataLoader:
        dataset = self._make_dataset(split)

        sampler = None
        if self.distributed_sampler:
            sampler = torch.utils.data.DistributedSampler(dataset, shuffle=shuffle)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self._make_loader("multicoil_train", shuffle=True)

    def val_dataloader(self):
        return self._make_loader("multicoil_val", shuffle=False)

    @staticmethod
    def add_data_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_path", default=None, type=Path, help="Path to fastMRI data root")
        parser.add_argument("--batch_size", default=1, type=int, help="Data loader batch size")
        parser.add_argument("--num_workers", default=4, type=int, help="Number of data loader workers")
        return parser
