from typing import Optional, Union
import logging
import math

import torch
import torch.utils
from torch.utils.data import Dataset
from torch import Tensor
from numpy import ndarray
import sklearn.cluster
import numpy as np

from ptychi.ptychotorch.utils import to_tensor, to_numpy

logger = logging.getLogger(__name__)


class PtychographyDataset(Dataset):
    def __init__(
        self,
        patterns: Union[Tensor, ndarray],
        valid_pixel_mask: Optional[Union[Tensor, ndarray]] = None,
        wavelength_m: float = None,
        propagation_distance_m: float = 1.0,
        fft_shift: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.patterns = to_tensor(patterns, device="cpu")
        if fft_shift:
            self.patterns = torch.fft.fftshift(self.patterns, dim=(-2, -1))
            logger.info("Diffraction data have been FFT-shifted.")

        if valid_pixel_mask is None:
            valid_pixel_mask = torch.ones(self.patterns.shape[-2:])
        self.valid_pixel_mask = to_tensor(valid_pixel_mask, device="cpu", dtype=torch.bool)

        self.wavelength_m = wavelength_m
        self.propagation_distance_m = propagation_distance_m

    def __getitem__(self, index):
        if not isinstance(index, torch.Tensor):
            index = torch.tensor(index, device="cpu", dtype=torch.long)
        pattern = self.patterns[index]
        return index, pattern

    def __len__(self):
        return len(self.patterns)

    def move_attributes_to_device(self, device=None):
        if device is None:
            device = torch.get_default_device()
        self.valid_pixel_mask = self.valid_pixel_mask.to(device)
        

class PtychographyCompactBatchSampler(torch.utils.data.Sampler):
    def __init__(self, positions, batch_size, *args, **kwargs):
        """
        A batch sampler that returns minibatches containing indices
        that are clustered together using k-Means. This is equivalent
        to the "compact" mode in PtychoShelves.

        Parameters
        ----------
        positions : Tensor
            A (N, 2) tensor of probe positions in pixels.
        batch_size : int
            The batch size.
        """
        self.positions = to_numpy(positions)
        self.batch_size = batch_size
        self.clusters_of_indices = []

        self.build_clusters()

    def build_clusters(self):
        self.clusters_of_indices = []
        kmeans = sklearn.cluster.KMeans(n_clusters=self.__len__())
        kmeans.fit(self.positions)
        cluster_labels = kmeans.predict(self.positions)
        for i_cluster in range(kmeans.n_clusters):
            self.clusters_of_indices.append(
                to_tensor(np.where(cluster_labels == i_cluster)[0], device="cpu", dtype=torch.long)
            )
            
    def update_clusters(self, positions):
        self.positions = to_numpy(positions)
        self.build_clusters()

    def __len__(self):
        return int(np.round(len(self.positions) / self.batch_size))

    def __iter__(self):
        cluster_indices = np.random.permutation(len(self.clusters_of_indices))
        for i_batch in cluster_indices:
            yield self.clusters_of_indices[i_batch]
