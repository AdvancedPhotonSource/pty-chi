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
import scipy.spatial

from ptychi.utils import to_tensor, to_numpy
import ptychi.maths as pmath

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
        to the "compact" mode in PtychoShelves. This implementation is
        adapted from `get_close_indices.m` in PtychoShelves.

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
        
    def __len__(self):
        if len(self.clusters_of_indices) == 0:
            raise ValueError("No clusters have been built yet. Call `build_clusters()` first.")
        return len(self.clusters_of_indices)

    def __iter__(self):
        for i_batch in range(len(self)):
            yield self.clusters_of_indices[i_batch]
            
    def estimate_n_clusters_iniitially(self):
        return int(np.ceil(len(self.positions) / self.batch_size))

    def build_clusters(self):
        self.clusters_of_indices = []
        cluster_labels, centers, _, distmat = self.get_best_kmeans()
        cluster_labels, centers, distmat = self.refine_clusters(cluster_labels, centers, distmat)
        
        # Sort cluster labels, so that the largest cluster is labeled as 0.
        n_clusters = len(centers)
        bin_populations, _ = np.histogram(cluster_labels, bins=n_clusters)
        sorted_inds = np.argsort(bin_populations)[::-1]
        
        # Populate batch indices.
        for i_cluster in range(n_clusters):
            self.clusters_of_indices.append(
                to_tensor(
                    np.where(cluster_labels == sorted_inds[i_cluster])[0], 
                    device="cpu", dtype=torch.long)
            )
        
        # PtychoShelves would truncate the clusters to the smallest batch size and throw
        # away some diffraction patterns if the range of cluster population is equal 
        # to or less than 1 to make it run faster on GPU. We don't do it here.
        return

    def update_clusters(self, positions):
        self.positions = to_numpy(positions)
        self.build_clusters()
        
    def get_best_kmeans(self, n_trials=10):
        """Run multiple k-Means trials and return the best one that has the
        most uniform cluster populations.
        
        Parameters
        ----------
        n_trials : int
            The number of k-Means trials to run.
            
        Returns
        -------
        ndarray
            An (n_positions,) array of cluster labels of the best run.
        ndarray
            An (n_clusters, 2) array of cluster centers of the best run.
        ndarray
            An (n_clusters, ) array giving the sum of squared distances of all
            points belonging to a cluster to the cluster center, for all
            clusters of the best run. 
        ndarray
            An (n_positions, n_clusters) matrix giving the squared distance of each
            point to each cluster center of the best run.
        """
        best_score = np.inf
        best_cluster_labels = None
        best_centers = None
        best_sumd = np.inf
        best_distmat = None
        for _ in range(n_trials):
            kmeans_engine = sklearn.cluster.KMeans(n_clusters=self.estimate_n_clusters_iniitially())
            kmeans_engine.fit(self.positions)
            cluster_labels = kmeans_engine.predict(self.positions)
            bin_populations = np.histogram(cluster_labels, bins=kmeans_engine.n_clusters)[0]
            score = np.std(bin_populations)
            if score < best_score:
                best_score = score
                best_cluster_labels = cluster_labels
                best_centers = kmeans_engine.cluster_centers_
                best_distmat = scipy.spatial.distance.cdist(self.positions, best_centers) ** 2
                best_sumd = self.get_within_cluster_distances(self.positions, best_cluster_labels, best_centers)
        return best_cluster_labels, best_centers, best_sumd, best_distmat
    
    def refine_clusters(self, cluster_labels, centers, distmat):
        iter = 0
        n_clusters = len(centers)
        n_pos = len(self.positions)
        
        adjusted_batch_size = int(np.ceil(n_pos / np.ceil(n_pos / self.batch_size)))
        
        # Iteratively homogenize the cluster populations by moving points belonging
        # to large clusters and are closest to the smallest cluster's center to the 
        # smallest cluster.
        while True:
            iter += 1
            bin_populations, _ = np.histogram(cluster_labels, bins=n_clusters)
            n_unique_populations = len(np.unique(bin_populations))
            if n_unique_populations == 1:
                break
            if n_unique_populations < max(2, math.ceil(iter / 1e3)) and (
                n_clusters * self.batch_size != n_pos or iter > 1e3
            ):
                break
            
            # Find the cluster with the smallest population and add new points to it.
            ind_smallest_cluster = np.argmin(bin_populations)
            inds_large_cluster = np.where(bin_populations > adjusted_batch_size)[0]
            if len(inds_large_cluster) == 0 or ind_smallest_cluster in inds_large_cluster:
                break
            # Find the index (indices) of the point(s) that
            # (1) belong to large clusters, and
            # (2) are closest to the smallest cluster.
            # We can have multiple such points if there is a tie in the distance.
            mask_points_in_large_clusters = np.isin(cluster_labels, inds_large_cluster)
            mask_points_closest_to_smallest_cluster = \
                distmat[:, ind_smallest_cluster] == np.min(
                    distmat[mask_points_in_large_clusters, ind_smallest_cluster]
                )
            # Assign these points to the smallest cluster.
            cluster_labels[mask_points_closest_to_smallest_cluster] = ind_smallest_cluster
            
        # Remove empty clusters. Labels of the new clusters are still consecutive.
        #         |<- points ->|
        #        | 0 1 0 0                                 0 0 0 0 
        # clusters 1 0 1 0 ...   =replace 1s with labels=> 1 0 1 0 ... =sum=> 1 0 1 2 ...
        #        | 0 0 0 1                                 0 0 0 2
        unique_cluster_labels = np.unique(cluster_labels)
        n_clusters = len(unique_cluster_labels)
        cluster_labels = np.sum(
            np.arange(n_clusters)[:, None] * (cluster_labels == unique_cluster_labels[:, None]), 
            axis=0
        )
        cluster_labels = cluster_labels.astype(int)
        
        # Rebuild cluster centers and distance matrix.
        cluster_centers = np.zeros((n_clusters, 2))
        for i_cluster in range(n_clusters):
            cluster_centers[i_cluster, :] = np.median(self.positions[cluster_labels == i_cluster], axis=0)
        distmat = scipy.spatial.distance.cdist(self.positions, cluster_centers) ** 2
        
        # Find more compact refinement.
        # For each point, get a list of cluster indices ranked by the distance 
        # between the point and the cluster centers.
        sind = np.argsort(distmat, axis=1)
        optimal_cluster_labels = sind[:, 0]
        
        # Iteratively find the points that are non-optimal (i.e., belong to a cluster whose
        # center is not the closest one) and swap them to the optimal cluster.
        largest_nonoptimal_ratio = 1
        for iter in range(10):
            mask_nonoptimal_points = cluster_labels != optimal_cluster_labels
            nonoptimal_ratio = np.mean(mask_nonoptimal_points)
            if nonoptimal_ratio > largest_nonoptimal_ratio:
                break
            largest_nonoptimal_ratio = nonoptimal_ratio
            
            for i in range(np.sum(mask_nonoptimal_points)):
                dists_to_cluster_centers = distmat[tuple(range(n_pos)), cluster_labels]
                ind_worst = pmath.masked_argmax(dists_to_cluster_centers, mask_nonoptimal_points)
                cluster_old = cluster_labels[ind_worst]
                cluster_new = optimal_cluster_labels[ind_worst]
                # Find the index of the point in cluster_new to swap with. This is the point
                # that is closest to the center of cluster_old.
                ind_new = pmath.masked_argmin(distmat[:, cluster_old], cluster_labels == cluster_new)
                
                cluster_labels[ind_worst] = cluster_new
                cluster_labels[ind_new] = cluster_old
        
                mask_nonoptimal_points[np.array([ind_worst, ind_new])] = 0
                if np.all(mask_nonoptimal_points == 0):
                    break
        return cluster_labels, cluster_centers, distmat

    @staticmethod
    def get_within_cluster_distances(positions, cluster_labels, centers):
        """Calculate the sum of squared distances of all points belonging to a cluster
        to the cluster center.
        
        Parameters
        ----------
        positions : ndarray
            A (n_positions, 2) array of probe positions in pixels.
        cluster_labels : ndarray
            An (n_positions,) array of cluster labels.
        centers : ndarray
            An (n_clusters, 2) array of cluster centers.
        """
        n_clusters = len(centers)
        sumds = np.zeros(n_clusters)
        for i_cluster in range(n_clusters):
            inds_of_cluster = np.where(cluster_labels == i_cluster)[0]
            sq_dists = np.sum((positions[inds_of_cluster] - centers[i_cluster]) ** 2, axis=1)
            sumds[i_cluster] = np.sum(sq_dists)
        return sumds


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
