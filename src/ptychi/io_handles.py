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

from ptychi.utils import to_tensor, to_numpy

logger = logging.getLogger(__name__)


class PtychographyDataset(Dataset):
    def __init__(
        self,
        patterns: Union[Tensor, ndarray],
        valid_pixel_mask: Optional[Union[Tensor, ndarray]] = None,
        wavelength_m: float = None,
        free_space_propagation_distance_m: float = 1.0,
        fft_shift: bool = True,
        save_data_on_device: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.patterns = to_tensor(patterns, device="cpu" if not save_data_on_device else "cuda")
        if fft_shift:
            self.patterns = torch.fft.fftshift(self.patterns, dim=(-2, -1))
            logger.info("Diffraction data have been FFT-shifted.")

        if valid_pixel_mask is None:
            valid_pixel_mask = torch.ones(self.patterns.shape[-2:])
        self.valid_pixel_mask = to_tensor(valid_pixel_mask, device="cpu", dtype=torch.bool)

        self.wavelength_m = wavelength_m
        self.free_space_propagation_distance_m = free_space_propagation_distance_m
        
        self.save_data_on_device = save_data_on_device

    def __getitem__(self, index):
        if not isinstance(index, torch.Tensor):
            index = torch.tensor(index, device=self.patterns.device, dtype=torch.long)
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
        return int(np.ceil(len(self.positions) / self.batch_size))

    def __iter__(self):
        cluster_indices = np.random.permutation(len(self.clusters_of_indices))
        for i_batch in cluster_indices:
            yield self.clusters_of_indices[i_batch]


class PtychographyUniformBatchSampler(torch.utils.data.Sampler):
    def __init__(self, positions, batch_size, *args, **kwargs):
        """A batch sampler that returns minibatches of indices that are random,
        but modified to be spread out over the space as uniformly as possible.
        This sampler is equivalent to the "sparse" mode in PtychoShelves.

        Parameters
        ----------
        positions : Tensor
            A (N, 2) tensor of probe positions in pixels.
        batch_size : int
            The batch size.
        """
        self.positions = positions
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.positions) / self.batch_size)

    def __iter__(self):
        self.build_indices()
        for i in range(len(self)):
            yield self.batches_of_indices[i]

    def build_indices(self):
        dist_mat = torch.cdist(self.positions, self.positions, p=2)
        dist_mat[dist_mat == 0] = torch.inf
        batches_of_indices = torch.split(
            torch.randperm(len(self.positions), device=dist_mat.device), self.batch_size
        )

        for i_batch in range(len(self) - 1):
            inds_current_batch = batches_of_indices[i_batch]
            dist_mat_current_batch = dist_mat[inds_current_batch, :][:, inds_current_batch]
            for i_pass in range(2):
                for ind_next_batch in range(len(batches_of_indices[i_batch + 1])):
                    # Calculate the inverse-distance weighted average of the distances to other
                    # points in the batch for every point:
                    # $s_i = \frac{1}{n} sum_j(d_{ij} * 1/d_{ij}) / sum_j(1/d_{ij}) = 1 / sum_j(1 / d_{ij})$
                    # Closer neighbors are more heavily
                    # weighted. This prevents the algorithm from always selecting the point
                    # around the center of the batch as in the case of a simple sum.
                    avgs_of_dists = 1.0 / torch.sum(1.0 / dist_mat_current_batch**2, dim=0)
                    if torch.all(torch.isinf(avgs_of_dists)):
                        break

                    # Swap the scan point with the smallest summed distance to other points in the batch
                    # with the scan point in the next batch.
                    min_dist_ind = torch.min(avgs_of_dists, dim=0).indices
                    (
                        batches_of_indices[i_batch][min_dist_ind],
                        batches_of_indices[i_batch + 1][ind_next_batch],
                    ) = (
                        batches_of_indices[i_batch + 1][ind_next_batch].item(),
                        batches_of_indices[i_batch][min_dist_ind].item(),
                    )

                    # Update distance matrix.
                    dist_mat_current_batch_update = dist_mat[
                        batches_of_indices[i_batch][min_dist_ind], batches_of_indices[i_batch]
                    ].clone()
                    dist_mat_current_batch[min_dist_ind, :] = dist_mat_current_batch_update
                    dist_mat_current_batch[:, min_dist_ind] = dist_mat_current_batch_update
        self.batches_of_indices = batches_of_indices
