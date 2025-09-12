# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Optional, Union, TYPE_CHECKING
import logging
import enum
import os

import numpy as np
import tifffile
import torch
from torch import Tensor

import ptychi.image_proc as ip
import ptychi.maths as pmath
import ptychi.utils as utils
import ptychi.data_structures.base as dsbase
import ptychi.data_structures.object as object
import ptychi.data_structures.opr_mode_weights as oprweights
from ptychi.propagate import FourierPropagator, WavefieldPropagator
import ptychi.maps as maps

if TYPE_CHECKING:
    import ptychi.api as api

logger = logging.getLogger(__name__)


class ProbeRepresentation(enum.StrEnum):
    NORMAL = enum.auto()
    SPARSE_CODE = enum.auto()
    DIP = enum.auto()


class Probe(dsbase.ReconstructParameter):
    # TODO: eigenmode_update_relaxation is only used for LSQML. We should create dataclasses
    # to contain additional options for ReconstructParameter classes, and subclass them for specific
    # reconstruction algorithms - for example, ProbeOptions -> LSQMLProbeOptions.
    options: "api.options.base.ProbeOptions"

    representation: ProbeRepresentation = ProbeRepresentation.NORMAL

    def __init__(
        self,
        name: str = "probe",
        options: "api.options.base.ProbeOptions" = None,
        *args,
        **kwargs,
    ):
        """
        Represents the probe function in a tensor of shape
            `(n_opr_modes, n_modes, h, w)`
        where:
        - n_opr_modes is the number of mutually coherent probe modes used in orthogonal
          probe relaxation (OPR).
        - n_modes is the number of mutually incoherent probe modes.
        """
        super().__init__(*args, name=name, options=options, is_complex=True, **kwargs)
        if len(self.shape) != 4:
            raise ValueError("Probe tensor must be of shape (n_opr_modes, n_modes, h, w).")

        self.probe_power = options.power_constraint.probe_power
        self.orthogonalize_incoherent_modes_method = options.orthogonalize_incoherent_modes.method

    def shift(self, shifts: Tensor):
        """
        Generate shifted probe.

        Parameters
        ----------
        shifts : Tensor
            A tensor of shape (2,) or (N, 2) giving the shifts in pixels.
            If a (N, 2)-shaped tensor is given, a batch of shifted probes are generated.

        Returns
        -------
        shifted_probe : Tensor
            Shifted probe.
        """
        if shifts.ndim == 1:
            probe_straightened = self.tensor.complex().view(-1, *self.shape[-2:])
            shifted_probe = ip.fourier_shift(
                probe_straightened, shifts[None, :].repeat([probe_straightened.shape[0], 1])
            )
            shifted_probe = shifted_probe.view(*self.shape)
        else:
            n_shifts = shifts.shape[0]
            n_images_each_probe = self.shape[0] * self.shape[1]
            probe_straightened = self.tensor.complex().view(n_images_each_probe, *self.shape[-2:])
            probe_straightened = probe_straightened.repeat(n_shifts, 1, 1)
            shifts = shifts.repeat_interleave(n_images_each_probe, dim=0)
            shifted_probe = ip.fourier_shift(probe_straightened, shifts)
            shifted_probe = shifted_probe.reshape(n_shifts, *self.shape)
        return shifted_probe

    @property
    def n_modes(self):
        return self.tensor.shape[1]

    @property
    def n_opr_modes(self):
        return self.tensor.shape[0]

    @property
    def has_multiple_opr_modes(self):
        return self.n_opr_modes > 1

    @property
    def lateral_shape(self):
        return self.shape[-2:]

    @property
    def has_multiple_incoherent_modes(self):
        return self.n_modes > 1

    def get_mode(self, mode: int, keepdim: bool = False):
        mode = self.tensor.complex()[:, mode]
        if keepdim:
            mode = mode.unsqueeze(1)
        return mode

    def get_opr_mode(self, mode: int, keepdim: bool = False):
        mode = self.tensor.complex()[mode]
        if keepdim:
            mode = mode.unsqueeze(0)
        return mode

    def get_mode_and_opr_mode(self, mode: int, opr_mode: int):
        return self.tensor.complex()[opr_mode, mode]

    def get_spatial_shape(self):
        return self.shape[-2:]

    def get_all_mode_intensity(
        self,
        opr_mode: Optional[int] = 0,
        weights: Optional[Union[Tensor, "dsbase.ReconstructParameter"]] = None,
    ) -> Tensor:
        """
        Get the intensity of all probe modes.

        Parameters
        ----------
        opr_mode : Optional[int]
            the OPR mode. If this is not None, only the intensity of the chosen
            OPR mode is calculated. Otherwise, it calculates the intensity of the weighted sum
            of all OPR modes. In that case, `weights` must be given.
        weights : Optional[Union[Tensor, ReconstructParameter]]
            a (n_opr_modes,) tensor giving the weights of OPR modes.

        Returns
        -------
        intensity : Tensor
            The summed intensity of all probe modes.
        """
        if isinstance(weights, oprweights.OPRModeWeights):
            weights = weights.data
        if opr_mode is not None:
            p = self.data[opr_mode]
        else:
            p = (self.data * weights[None, :, :, :]).sum(0)
        return torch.sum((p.abs()) ** 2, dim=0)

    def get_unique_probes(
        self,
        weights: Union[Tensor, "dsbase.ReconstructParameter"],
        mode_to_apply: Optional[int] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        weights : Tensor
            A tensor giving the weights of the eigenmodes.
        mode_to_apply : int, optional
            The incoherent mode for which OPR modes should be applied. The data
            for other modes will be set to the value of the first OPR mode. If None,
            OPR correction will be done to all incoherent modes.

        Returns
        -------
        unique_probe : Tensor
            A tensor of unique probes. If weights.ndim == 2, the shape is (n_points, n_modes, h, w).
            If weights.ndim == 1, the shape is (n_modes, h, w).
        """
        if isinstance(weights, oprweights.OPRModeWeights):
            weights = weights.data

        p_orig = None
        if mode_to_apply is not None:
            p_orig = self.data.clone()
            p = p_orig[:, [mode_to_apply], :, :]
        else:
            p = self.data.clone()
        if weights.ndim == 1:
            unique_probe = p * weights[:, None, None, None]
            unique_probe = unique_probe.sum(0)
        else:
            unique_probe = p[None, ...] * weights[:, :, None, None, None]
            unique_probe = unique_probe.sum(1)

        # If OPR is only applied on one incoherent mode, add in the rest of the modes.
        if mode_to_apply is not None:
            if weights.ndim == 1:
                # Shape of unique_probe:     (1, h, w)
                p_orig[0, [mode_to_apply], :, :] = unique_probe
                unique_probe = p_orig[0, ...]
            else:
                # Shape of unique_probe:     (n_points, 1, h, w)
                p_orig = p_orig[0:1, ...].repeat(weights.shape[0], 1, 1, 1)
                p_orig[:, [mode_to_apply], :, :] = unique_probe
                unique_probe = p_orig
        return unique_probe

    def _get_probe_mode_slicer(self, mode_index=None):
        if mode_index is None:
            return slice(None)
        else:
            return slice(mode_index, mode_index + 1)

    def constrain_incoherent_modes_orthogonality(self):
        """Orthogonalize the incoherent probe modes for the first OPR mode."""
        if not self.has_multiple_incoherent_modes:
            return

        probe = self.data

        if self.options.orthogonalize_incoherent_modes.sort_by_occupancy:
            shared_occupancy = torch.sum(torch.abs(probe[0, ...]) ** 2, (-2, -1)) / torch.sum(
                torch.abs(probe[0, ...]) ** 2
            )
            shared_occupancy = torch.sort(shared_occupancy, dim=0, descending=True)
            probe[0, ...] = probe[0, shared_occupancy[1], ...]

        norm_first_mode_orig = pmath.norm(probe[0, 0], dim=(-2, -1))

        if self.orthogonalize_incoherent_modes_method == "gs":
            func = pmath.orthogonalize_gs
        elif self.orthogonalize_incoherent_modes_method == "svd":
            func = pmath.orthogonalize_svd
        else:
            raise NotImplementedError(
                f"Orthogonalization method {self.orthogonalize_incoherent_modes_method} "
                "is not supported."
            )
        probe[0] = func(
            probe[0],
            dim=(-2, -1),
            group_dim=0,
        )

        # Restore norm.
        if self.orthogonalize_incoherent_modes_method != "svd":
            norm_first_mode_new = pmath.norm(probe[0, 0], dim=(-2, -1))
            probe = probe * norm_first_mode_orig / norm_first_mode_new

        self.set_data(probe)

    def constrain_opr_mode_orthogonality(
        self, weights: "oprweights.OPRModeWeights", eps=1e-5, *, update_weights_in_place: bool
    ):
        """
        Add the following constraints to variable probe weights

        1. Remove outliars from weights
        2. Enforce orthogonality once per epoch
        3. Sort the variable probes by their total energy
        4. Normalize the variable probes so the energy is contained in the weight

        Adapted from Tike (https://github.com/AdvancedPhotonSource/tike). The implementation
        in Tike assumes a separately stored variable probe eigenmodes; here we use the
        PtychoShelves convention and regard the second and following OPR modes as eigenmodes.

        Also, this function assumes that OPR correction is only applied to the first
        incoherent mode when mixed state probe is used, as this is what PtychoShelves does.
        OPR modes of other incoherent modes are ignored, for now.

        This function also updates the OPR mode weights when `update_weights_in_place` is True.

        Parameters
        ----------
        weights : OPRModeWeights
            The OPR mode weights object.
        update_weights_in_place : bool
            Whether to update the OPR mode weights data.

        Returns
        -------
        Tensor, optional
            Normalized and sorted OPR mode weights.
        """
        if not self.has_multiple_opr_modes:
            return

        weights_data = weights.data

        # The main mode of the probe is the first OPR mode, while the
        # variable part of the probe is the second and following OPR modes.
        # The main mode should not change during orthogonalization, but the
        # variable OPR modes should all be orthogonal to it.
        probe = self.data

        # TODO: remove outliars by polynomial fitting (remove_variable_probe_ambiguities.m)

        # Normalize eigenmodes and adjust weights.
        eigenmodes = probe[1:, ...]
        vnorm = pmath.mnorm(eigenmodes, dim=(-2, -1), keepdims=True)
        eigenmodes /= vnorm + eps
        # Shape of weights:      (n_points, n_opr_modes).
        # Currently, only the first incoherent mode has OPR modes, and the
        # stored weights are for that mode.
        weights_data[:, 1:] = weights_data[:, 1:] * vnorm[:, 0, 0, 0]

        # Orthogonalize variable probes. With Gram-Schmidt, the first
        # OPR mode (i.e., the main mode) should not change during orthogonalization.
        probe = pmath.orthogonalize_gs(
            probe,
            dim=(-2, -1),
            group_dim=0,
        )

        if False:
            # Compute the energies of variable OPR modes (i.e., the second and following)
            # in order to sort probes by energy.
            # Shape of power:         (n_opr_modes - 1,).
            power = pmath.norm(weights_data[..., 1:], dim=0) ** 2

            # Sort the probes by energy
            sorted = torch.argsort(-power)
            weights_data[:, 1:] = weights_data[:, sorted + 1]
            # Apply only to the first incoherent mode.
            probe[1:, 0, :, :] = probe[sorted + 1, 0, :, :]

        # Remove outliars from variable probe weights.
        aevol = torch.abs(weights_data)
        weights_data = torch.minimum(
            aevol,
            1.5
            * torch.quantile(
                aevol,
                0.95,
                dim=0,
                keepdims=True,
            ).type(weights_data.dtype),
        ) * torch.sign(weights_data)

        # Update stored data.
        self.set_data(probe)

        if update_weights_in_place:
            weights.set_data(weights_data)
        else:
            return weights_data

    def constrain_probe_power(
        self,
        object_: "object.Object",
        opr_mode_weights: Union[Tensor, "oprweights.OPRModeWeights"],
        propagator: Optional[WavefieldPropagator] = None,
    ) -> None:
        if self.probe_power <= 0.0:
            return

        if isinstance(opr_mode_weights, oprweights.OPRModeWeights):
            opr_mode_weights = opr_mode_weights.data

        if propagator is None:
            propagator = FourierPropagator()

        # Shape of probe_composed:        (n_modes, h, w)
        if self.has_multiple_opr_modes:
            avg_weights = opr_mode_weights.mean(dim=0)
            probe_composed = self.get_unique_probes(avg_weights, mode_to_apply=0)
        else:
            probe_composed = self.get_opr_mode(0)

        propagated_probe = propagator.propagate_forward(probe_composed)
        if isinstance(propagator, FourierPropagator):
            # Cancel the normalization factor so that the power is conserved.
            if propagator.norm == "backward" or propagator.norm is None:
                propagated_probe_power = (
                    torch.sum(propagated_probe.abs() ** 2) / self.data.size().numel()
                )
            elif propagator.norm == "forward":
                propagated_probe_power = (
                    torch.sum(propagated_probe.abs() ** 2) * self.data.size().numel()
                )
            else:
                propagated_probe_power = torch.sum(propagated_probe.abs() ** 2)
        else:
            propagated_probe_power = torch.sum(propagated_probe.abs() ** 2)
        power_correction = torch.sqrt(self.probe_power / propagated_probe_power)

        self.set_data(self.data * power_correction)
        object_.set_data(object_.data / power_correction)

        logger.info("Probe and object scaled by {}.".format(power_correction))

    def constrain_support(self):
        """
        Apply probe support constraint through a mask generated by thresholding the
        blurred probe magnitude.
        """
        data = self.data
        mask = ip.gaussian_filter(data, sigma=3, size=5).abs()
        thresh = (
            mask.max(-1, keepdim=True).values.max(-2, keepdim=True).values
            * self.options.support_constraint.threshold
        )
        mask = torch.where(mask > thresh, 1.0, 0.0)
        mask = ip.gaussian_filter(mask, sigma=2, size=3).abs()
        data = data * mask
        self.set_data(data)

    def center_probe(self):
        """
        Move the probe's center of mass to the center of the probe array.
        """

        if self.options.center_constraint.use_intensity_for_com:
            probe_to_be_shifted = torch.sum(torch.abs(self.data[0, ...]) ** 2, dim=0)
        else:
            probe_to_be_shifted = self.get_mode_and_opr_mode(0, 0)

        com = ip.find_center_of_mass(probe_to_be_shifted)
        shift = utils.to_tensor(self.shape[-2:]) // 2 - com
        shifted_probe = self.shift(shift)
        self.set_data(shifted_probe)

    def post_update_hook(self) -> None:
        super().post_update_hook()

    def normalize_eigenmodes(self):
        """
        Normalize all eigenmodes (the second and following OPR modes) such that each of them
        has a squared norm equal to the number of pixels in the probe.
        """
        if not self.has_multiple_opr_modes:
            return
        eigen_modes = self.data[1:, ...]
        for i_opr_mode in range(eigen_modes.shape[0]):
            for i_mode in range(eigen_modes.shape[1]):
                eigen_modes[i_opr_mode, i_mode, :, :] /= (
                    pmath.mnorm(eigen_modes[i_opr_mode, i_mode, :, :], dim=(-2, -1)) + 1e-8
                )

        new_data = self.data
        new_data[1:, ...] = eigen_modes
        self.set_data(new_data)

    def save_tiff(self, path: str):
        """
        Save the probe's magnitude and phase as 2 TIFF files. Each file contains
        an array of tiles, where the rows correspond to incoherent probe modes
        and columns correspond to OPR modes.

        Parameters
        ----------
        path : str
            Path to save. "_phase" and "_mag" will be appended to the filename.
        """
        fname = os.path.splitext(path)[0]
        mag_img = np.empty([self.shape[3] * self.shape[1], self.shape[2] * self.shape[0]])
        phase_img = np.empty([self.shape[3] * self.shape[1], self.shape[2] * self.shape[0]])
        data = self.data
        for i_mode in range(self.shape[1]):
            for i_opr_mode in range(self.shape[0]):
                mag_img[
                    i_mode * self.shape[3] : (i_mode + 1) * self.shape[3],
                    i_opr_mode * self.shape[2] : (i_opr_mode + 1) * self.shape[2],
                ] = data[i_opr_mode, i_mode, :, :].abs().detach().cpu().numpy()
                phase_img[
                    i_mode * self.shape[3] : (i_mode + 1) * self.shape[3],
                    i_opr_mode * self.shape[2] : (i_opr_mode + 1) * self.shape[2],
                ] = torch.angle(data[i_opr_mode, i_mode, :, :]).detach().cpu().numpy()
        tifffile.imwrite(fname + "_mag.tif", mag_img)
        tifffile.imwrite(fname + "_phase.tif", phase_img)


class SynthesisDictLearnProbe(Probe):
    representation: ProbeRepresentation = ProbeRepresentation.SPARSE_CODE

    def __init__(self, name="probe", options=None, *args, **kwargs):
        super().__init__(
            name, options, build_optimizer=False, data_as_parameter=False, *args, **kwargs
        )

        dictionary_matrix, dictionary_matrix_pinv = self.get_dictionary()
        self.register_buffer("dictionary_matrix", dictionary_matrix)
        self.register_buffer("dictionary_matrix_pinv", dictionary_matrix_pinv)

        sparse_code_probe_shared_nnz = torch.tensor(
            self.options.experimental.sdl_probe_options.sparse_code_probe_shared_nnz,
            dtype=torch.uint32,
        )
        sparse_code_probe_shared = self.get_sparse_code_probe_shared_weights()
        self.register_buffer("sparse_code_probe_shared_nnz", sparse_code_probe_shared_nnz)
        self.register_parameter(
            "sparse_code_probe_shared", torch.nn.Parameter(sparse_code_probe_shared)
        )

        self.build_optimizer()

    def get_dictionary(self):
        dictionary_matrix = torch.tensor(
            self.options.experimental.sdl_probe_options.dictionary_matrix, dtype=torch.complex64
        )
        dictionary_matrix_pinv = torch.tensor(
            self.options.experimental.sdl_probe_options.dictionary_matrix_pinv,
            dtype=torch.complex64,
        )
        return dictionary_matrix, dictionary_matrix_pinv

    def get_sparse_code_weights_vs_scanpositions(self, probe_vs_scanpositions):
        """Get the sparse code weights for a given probe vs scan positions.

        Parameters
        ----------
        probe_vs_scanpositions : Tensor
            A (n_pos, 1, h, w) tensor giving the probe vs scan positions.

        Returns
        -------
        Tensor
            A tensor giving the sparse code weights for the given probe vs scan positions.
        """
        sz = probe_vs_scanpositions.shape
        probe_vec = torch.reshape(probe_vs_scanpositions, (sz[0], sz[1], sz[2] * sz[3]))
        sparse_code_vs_scanpositions = torch.einsum(
            "ij,klj->ikl", self.dictionary_matrix_pinv, probe_vec
        )

        return sparse_code_vs_scanpositions

    def get_sparse_code_probe_shared_weights(self):
        probe_shared = self.data[0, ...]
        sz = probe_shared.shape
        probe_vec = torch.reshape(probe_shared, (sz[0], sz[1] * sz[2]))
        sparse_code_probe_shared = self.dictionary_matrix_pinv @ probe_vec.T

        return sparse_code_probe_shared.T

    def generate(self):
        """Generate the probe using the sparse code, and set the
        generated probe to self.data.

        Returns
        -------
        Tensor
            A (n_opr_modes, n_modes, h, w) tensor giving the generated probe.
        """

        if self.options.experimental.sdl_probe_options.enabled_shared:
            sz = self.data.shape
            probe = torch.zeros(*[sz], dtype=torch.complex64)

            probe_shared = self.dictionary_matrix @ self.sparse_code_probe_shared.T

            probe[0, ...] = torch.reshape(probe_shared.T, *[sz[1:]])
            probe[1:, 0, ...] = self.data[1:, 0, ...]

            self.set_data(probe)

        else:
            probe = self.data

    def build_optimizer(self):
        if self.optimizable and self.optimizer_class is None:
            raise ValueError(
                "Parameter {} is optimizable but no optimizer is specified.".format(self.name)
            )
        if self.optimizable:
            self.optimizer = self.optimizer_class(
                [self.sparse_code_probe_shared], **self.optimizer_params
            )

    def set_sparse_code_probe_shared(self, data):
        """
        Set the sparse code weights for the shared probe.

        Parameters
        ----------
        data : Tensor
            A (n_dict_bases, n_modes) tensor giving the sparse code weights for the shared probe.
        """
        self.sparse_code_probe_shared.data = data

    def set_sparse_code_weights_vs_scanpositions(
        self, sparse_code_vs_scanpositions: Tensor, indices: tuple | Tensor = None
    ):
        """
        Set the sparse code weights for a given probe vs scan positions.

        Parameters
        ----------
        sparse_code_vs_scanpositions : Tensor
            A (n_pos, n_opr_modes, n_scpm) tensor giving the sparse code weights for the
            given probe vs scan positions.
        indices : tuple | Tensor
            The indices to apply to the sparse code weights.
        """
        raise NotImplementedError("This method is not implemented yet.")
        if indices is None:
            indices = slice(None)
        self.sparse_code_weights_vs_scanpositions[indices] = sparse_code_vs_scanpositions

    def get_probe_update_direction_sparse_code_probe_shared(self, delta_p_i, chi, obj_patches):
        nr = chi.shape[-2]
        nc = chi.shape[-1]
        nrnc = nr * nc
        n_scpm = chi.shape[-3]
        n_spos = chi.shape[-4]

        obj_patches = torch.reshape(obj_patches, (n_spos, nrnc))
        chi = torch.reshape(chi, (n_spos, n_scpm, nrnc)).permute(2, 0, 1)

        # get sparse code update direction
        delta_sparse_code = torch.einsum(
            "ijk,kl->lij",
            torch.reshape(delta_p_i, (n_spos, n_scpm, nrnc)),
            self.dictionary_matrix.conj(),
        )

        # compute optimal step length for sparse code update
        dict_delta_sparse_code = torch.einsum(
            "ij,jkl->ikl", self.dictionary_matrix, delta_sparse_code
        )

        denom = (torch.abs(dict_delta_sparse_code) ** 2) * obj_patches.swapaxes(0, -1)[..., None]
        denom = torch.einsum("ij,jik->ik", torch.conj(obj_patches), denom)

        numer = torch.conj(dict_delta_sparse_code) * chi
        numer = torch.einsum("ij,jik->ik", torch.conj(obj_patches), numer)

        # real is used to throw away small imag part due to numerical precision errors
        optimal_step_sparse_code = (numer / denom).real

        optimal_delta_sparse_code = optimal_step_sparse_code[None, ...] * delta_sparse_code

        # enforce sparsity constraint on sparse code
        abs_sparse_code = torch.abs(optimal_delta_sparse_code)
        abs_sparse_code_sorted = torch.sort(abs_sparse_code, dim=0, descending=True)

        sel = abs_sparse_code_sorted[0][self.sparse_code_probe_shared_nnz, ...]
        sparse_code_mask = abs_sparse_code >= sel[None, ...]

        # hard or soft thresholding
        if self.options.experimental.sdl_probe_options.thresholding_type_shared == "hard":
            optimal_delta_sparse_code = optimal_delta_sparse_code * sparse_code_mask
        elif self.options.experimental.sdl_probe_options.thresholding_type_shared == "soft":
            optimal_delta_sparse_code = (
                (abs_sparse_code - sel[None, ...])
                * sparse_code_mask
                * torch.exp(1j * torch.angle(optimal_delta_sparse_code))
            )

        # update the shared probe sparse codes using the average over scan positions
        sparse_code_probe_shared = self.get_sparse_code_probe_shared_weights()
        sparse_code_probe_shared = sparse_code_probe_shared + optimal_delta_sparse_code.mean(1).T
        self.set_sparse_code_probe_shared(sparse_code_probe_shared)

        delta_p_i = torch.einsum(
            "ij,jlk->ilk", self.dictionary_matrix, optimal_delta_sparse_code
        ).permute(1, 2, 0)

        delta_p_i = torch.reshape(delta_p_i, (n_spos, n_scpm, nr, nc))

        return delta_p_i


class DIPProbe(Probe):
    options: "api.options.ad_ptychography.AutodiffPtychographyProbeOptions"
    representation: ProbeRepresentation = ProbeRepresentation.DIP

    def __init__(
        self,
        name: str = "probe",
        options: "api.options.ad_ptychography.AutodiffPtychographyProbeOptions" = None,
        *args,
        **kwargs,
    ) -> None:
        """Deep image prior object.

        Parameters
        ----------
        name : str, optional
            The name of the object.
        options : api.options.base.ObjectOptions, optional
            The options for the object.
        """
        super().__init__(name=name, options=options, data_as_parameter=False, *args, **kwargs)
        self.model = None
        self.dip_output_magnitude = None
        self.dip_output_phase = None

        self.build_model()
        self.build_dip_optimizer()

        # `self.tensor` is used to hold the object generated by the DIP model and
        # is not trainable.
        self.tensor.requires_grad_(False)

        nn_input = self.get_nn_input()
        self.register_buffer("nn_input", nn_input)

        self.initial_data = None
        if self.options.experimental.deep_image_prior_options.residual_generation:
            self.initial_data = self.data.clone()

    def build_model(self):
        if not self.options.experimental.deep_image_prior_options.enabled:
            return
        model_class = maps.get_nn_model_by_enum(
            self.options.experimental.deep_image_prior_options.model
        )
        self.model = model_class(**self.options.experimental.deep_image_prior_options.model_params)

    def build_dip_optimizer(self):
        if self.optimizable and self.optimizer_class is None:
            raise ValueError(
                "Parameter {} is optimizable but no optimizer is specified.".format(self.name)
            )
        if self.optimizable:
            self.optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_params)

    def get_nn_input(self):
        z = (
            torch.rand(
                [
                    self.n_opr_modes * self.n_modes,
                    self.options.experimental.deep_image_prior_options.net_input_channels,
                    *self.lateral_shape,
                ],
            )
            * 0.1
        )
        return z

    def generate(self) -> Tensor:
        """Generate the probe using the deep image prior model, and set the
        generated probe to self.data.

        Returns
        -------
        Tensor
            The generated probe.
        """
        if self.model is None:
            raise ValueError("Model is not built.")
        p = self.model(self.nn_input)

        p, mag, phase = self.process_net_output(p)

        with torch.no_grad():
            self.dip_output_magnitude = mag.clone()
            self.dip_output_phase = phase.clone()

        if self.options.experimental.deep_image_prior_options.residual_generation:
            init_data = torch.stack([self.initial_data.real, self.initial_data.imag], dim=-1)
            p = p + init_data
        self.tensor.data = p
        return self.data

    def process_net_output(self, p):
        """Apply constraints to the DIP network's output and transform the tensor.

        Parameters
        ----------
        o : Tensor | tuple[Tensor, Tensor]
            The output of the DIP network. It should either be a [n_modes * n_opr_modes, 2, h, w]
            tensor with the channels giving the magnitude and phase of the probe,
            or a tuple of two [n_modes * n_opr_modes, h, w] tensors giving the magnitude and phase
            of the probe.

        Returns
        -------
        Tensor
            A [n_opr_modes, n_modes, h, w, 2] tensor representing the real and imaginary parts
            of the probe.
        Tensor
            The magnitude of the probe.
        Tensor
            The phase of the probe.
        """
        if isinstance(p, (list, tuple)):
            mag, phase = p
            mag = mag[:, 0]
            phase = phase[:, 0]
        else:
            mag = p[:, 0]
            phase = p[:, 1]

        expected_phase_shape = (self.n_opr_modes * self.n_modes, *self.lateral_shape)
        if tuple(phase.shape) != expected_phase_shape:
            logger.warning(
                "Shape of phase output by NN is expected to be "
                f"(n_slices, h, w) = {expected_phase_shape} but got {phase.shape}. "
                "Padding/cropping generated object to match expected shape."
            )
            mag_resized, phase_resized = [], []
            for i_img in range(expected_phase_shape[0]):
                mag_resized.append(ip.central_crop_or_pad(mag[i_img], expected_phase_shape[1:]))
                phase_resized.append(ip.central_crop_or_pad(phase[i_img], expected_phase_shape[1:]))
            mag = torch.stack(mag_resized)
            phase = torch.stack(phase_resized)

        p_complex = mag * torch.exp(1j * phase)
        p = torch.stack([p_complex.real, p_complex.imag], dim=-1)
        p = p.reshape([*self.shape, 2])
        return p, mag, phase
