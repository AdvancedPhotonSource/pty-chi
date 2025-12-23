# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Tuple, TYPE_CHECKING, Optional
import logging
import copy
import math

import torch
from torch import Tensor

import ptychi.image_proc as ip
import ptychi.data_structures as ds
import ptychi.data_structures.base as dsbase
import ptychi.maths as pmath
from ptychi.timing.timer_utils import timer
import ptychi.api.enums as enums
from ptychi.utils import (
    get_default_complex_dtype, 
    to_numpy, 
    chunked_processing,
    get_probe_renormalization_factor
)
import ptychi.maps as maps

if TYPE_CHECKING:
    import ptychi.api as api
    from ptychi.data_structures.probe_positions import ProbePositions
    
logger = logging.getLogger(__name__)


class SliceSpacings(dsbase.ReconstructParameter):
    options: "api.options.base.SliceSpacingOptions"

    def __init__(
        self,
        *args,
        name: str = "slice_spacings",
        options: "api.options.base.SliceSpacingOptions" = None,
        **kwargs,
    ):
        """
        Slice spacings.

        Parameters
        ----------
        data: a tensor of shape (n_slices,) giving the slice spacings in meters.
        """
        super().__init__(*args, name=name, options=options, is_complex=False, **kwargs)


class Object(dsbase.ReconstructParameter):
    options: "api.options.base.ObjectOptions"

    pixel_size_m: float = 1.0

    def __init__(
        self,
        name: str = "object",
        options: "api.options.base.ObjectOptions" = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, name=name, options=options, is_complex=True, **kwargs)
        self.pixel_size_m = options.pixel_size_m
        self.roi_bbox: dsbase.BoundingBox = None
        
        pos_origin_coords = torch.tensor(self.shape, device=torch.get_default_device()) / 2.0
        pos_origin_coords = pos_origin_coords.round() + 0.5
        self.register_buffer("pos_origin_coords", pos_origin_coords)

    def extract_patches(self, positions, patch_shape, *args, **kwargs):
        raise NotImplementedError

    def place_patches(self, positions, patches, *args, **kwargs):
        raise NotImplementedError

    @timer()
    def constrain_l1_norm(self):
        if self.options.l1_norm_constraint.weight <= 0:
            return
        data = self.data
        l1_grad = torch.sgn(data) / data.numel()
        data = data - self.options.l1_norm_constraint.weight * l1_grad
        self.set_data(data)
        logger.debug("L1 norm constraint applied to object.")
        
    def constrain_l2_norm(self):
        if self.options.l2_norm_constraint.weight <= 0:
            return
        data = self.data
        l2_grad = data * 2 / data.numel()
        data = data - self.options.l2_norm_constraint.weight * l2_grad
        self.set_data(data)
        logger.debug("L2 norm constraint applied to object.")

    def constrain_smoothness(self):
        raise NotImplementedError

    def constrain_total_variation(self) -> None:
        raise NotImplementedError

    def remove_grid_artifacts(self, *args, **kwargs):
        raise NotImplementedError
    
    def build_roi_bounding_box(self, positions: "ProbePositions"):
        pos = positions.data
        self.roi_bbox = dsbase.BoundingBox(
            sy=pos[:, 0].min(),
            ey=pos[:, 0].max(),
            sx=pos[:, 1].min(),
            ex=pos[:, 1].max(),
            origin=tuple(to_numpy(self.pos_origin_coords)),
        )
    
    def get_object_in_roi(self):
        raise NotImplementedError
    
    def update_preconditioner(self):
        raise NotImplementedError
    
    def initialize_preconditioner(self):
        raise NotImplementedError
    
    def remove_object_probe_ambiguity(self):
        raise NotImplementedError
    
    def update_pos_origin_coordinates(self, *args, **kwargs):
        raise NotImplementedError


class PlanarObject(Object):
    """
    Object that consists of one or multiple planes, i.e., 2D object or
    multislice object. The object is stored in a (n_slices, h , w) tensor.
    """

    def __init__(
        self,
        name: str = "object",
        options: "api.options.base.ObjectOptions" = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(name=name, options=options, *args, **kwargs)

        if len(self.shape) != 3:
            raise ValueError("PlanarObject should have a shape of (n_slices, h, w).")
        if (
            self.options.slice_spacings_m is not None
            and len(self.options.slice_spacings_m) != self.n_slices - 1
            and self.n_slices > 1
        ):
            raise ValueError("The number of slice spacings must be n_slices - 1.")

        # Build slice spacing object.
        if self.is_multislice:
            if self.options.slice_spacings_m is None:
                raise ValueError("slice_spacings_m must be specified for multislice objects.")
            slice_spacings_m = self.options.slice_spacings_m
        else:
            slice_spacings_m = torch.zeros(1)
            
        self.slice_spacings = SliceSpacings(
            data=slice_spacings_m,
            options=self.options.slice_spacing_options
        )
        self.register_optimizable_sub_module(self.slice_spacings)

        # Initialize position origin coordinates.
        pos_origin_coords = torch.tensor(self.shape[1:], device=torch.get_default_device()) / 2.0
        pos_origin_coords = pos_origin_coords.round() + 0.5
        self.register_buffer("pos_origin_coords", pos_origin_coords)

    @property
    def is_multislice(self) -> bool:
        return self.shape[0] > 1

    @property
    def n_slices(self):
        return self.shape[0]

    @property
    def lateral_shape(self):
        return self.shape[1:]

    @property
    def extract_patches_function(self) -> "ip.ExtractPatchesProtocol":
        return maps.get_patch_extractor_function_by_name(self.options.patch_interpolation_method)

    @property
    def place_patches_function(self) -> "ip.PlacePatchesProtocol":
        return maps.get_patch_placer_function_by_name(self.options.patch_interpolation_method)
    
    def update_pos_origin_coordinates(self, positions: Tensor = None):
        """Update the coordinates of the center pixel based on probe positions.

        Parameters
        ----------
        positions : Tensor
            A (n_pos, 2) tensor of probe positions in pixels.
        """
        if self.options.determine_position_origin_coords_by == enums.ObjectPosOriginCoordsMethods.POSITIONS:
            positions = positions.detach()
            buffer_center = torch.tensor([x / 2 for x in self.lateral_shape], device=positions.device)
            position_center = (positions.max(0).values + positions.min(0).values) / 2
            self.pos_origin_coords = buffer_center - position_center
            self.pos_origin_coords = self.pos_origin_coords.round() + 0.5
        elif self.options.determine_position_origin_coords_by == enums.ObjectPosOriginCoordsMethods.SUPPORT:
            pos_origin_coords = torch.tensor(self.shape[1:], device=torch.get_default_device()) / 2.0
            pos_origin_coords = pos_origin_coords.round() + 0.5
            self.pos_origin_coords = pos_origin_coords
        elif self.options.determine_position_origin_coords_by == enums.ObjectPosOriginCoordsMethods.SPECIFIED:
            if self.options.position_origin_coords is None:
                raise ValueError(
                    "`object_options.center_coords` should be specified when "
                    "`object_options.determine_center_coords_by` is set to "
                    "`SPECIFIED`."
                )
            self.pos_origin_coords = self.options.position_origin_coords
        else:
            raise ValueError(f"Invalid value for `determine_center_coords_by`: {self.options.determine_position_origin_coords_by}")

    def get_slice(self, index):
        return self.data[index, ...]

    @timer()
    def extract_patches(
        self, 
        positions: Tensor, 
        patch_shape: Tuple[int, int], 
        integer_mode: bool = False,
        pad_for_shift: Optional[int] = 1,
    ):
        """
        Extract (n_patches, n_slices, h', w') patches from the object.

        Parameters
        ----------
        positions : Tensor
            Tensor of shape (N, 2) giving the center positions of the patches in pixels.
            The origin of the given positions are assumed to be `self.pos_origin_coords`.
        patch_shape : tuple
            Tuple giving the lateral patch shape in pixels.
        integer_mode : bool, optional
            If True, the patches are extracted at the exact center between pixels,
            so that no interpolation is needed.
        pad_for_shift : int, optional
            If given, patches larger than the intended size by this amount are cropped
            out from the object before shifting.

        Returns
        -------
        Tensor
            Tensor of shape (N, n_slices, h', w') containing the extracted patches.
        """
        # Positions are provided with the origin in the center of the object support.
        # We shift the positions so that the origin is in the upper left corner.
        positions = positions + self.pos_origin_coords
        patches_all_slices = []
        for i_slice in range(self.n_slices):
            if integer_mode:
                patches = ip.extract_patches_integer(
                    self.get_slice(i_slice), positions, patch_shape
                )
            else:
                patches = self.extract_patches_function(
                    self.get_slice(i_slice), positions, patch_shape, pad=pad_for_shift
                )
            patches_all_slices.append(patches)
        patches_all_slices = torch.stack(patches_all_slices, dim=1)
        return patches_all_slices

    @timer()
    def place_patches(
        self, 
        positions: Tensor, 
        patches: Tensor, 
        integer_mode: bool = False,
        pad_for_shift: Optional[int] = 1,
        *args, **kwargs
    ):
        """
        Place patches into the object.

        Parameters
        ----------
        positions : Tensor
            Tensor of shape (N, 2) giving the center positions of the patches in pixels.
            The origin of the given positions are assumed to be `self.pos_origin_coords`.
        patches : Tensor
            Tensor of shape (n_patches, n_slices, H, W) of image patches.
        integer_mode : bool, optional
            If True, the patches are placed at the exact center between pixels,
            so that no interpolation is needed.
        pad_for_shift : int, optional
            If given, patches are either padded (for adjoint mode) or cropped 
            (for forward mode) before applying subpixel shifts.
        """
        positions = positions + self.pos_origin_coords
        updated_slices = []
        for i_slice in range(self.n_slices):
            if integer_mode:
                image = ip.place_patches_integer(
                    self.get_slice(i_slice), positions, patches[:, i_slice, ...], op="add"
                )
            else:
                image = self.place_patches_function(
                    self.get_slice(i_slice), positions, patches[:, i_slice, ...], op="add",
                    pad=pad_for_shift,
                )
            updated_slices.append(image)
        updated_slices = torch.stack(updated_slices, dim=0)
        self.tensor.set_data(updated_slices)

    @timer()
    def place_patches_on_empty_buffer(
        self, 
        positions: Tensor, 
        patches: Tensor, 
        integer_mode: bool = False,
        pad_for_shift: Optional[int] = 1,
        *args, **kwargs
    ):
        """
        Place patches into a zero array with the *lateral* shape of the object.

        Parameters
        ----------
        positions : Tensor
            Tensor of shape (N, 2) giving the center positions of the patches in pixels.
            The origin of the given positions are assumed to be `self.pos_origin_coords`.
        patches : Tensor
            Tensor of shape (N, H, W) of image patches.
        integer_mode : bool, optional
            If True, given positions are assumed to be integer-valued, so that no 
            interpolation is needed.
        pad_for_shift : int, optional
            If given, patches are either padded (for adjoint mode) or cropped 
            (for forward mode) before applying subpixel shifts.

        Returns
        -------
        image : Tensor
            A tensor with the lateral shape of the object with patches added onto it.
        """
        positions = positions + self.pos_origin_coords
        image = torch.zeros(
            self.lateral_shape, dtype=get_default_complex_dtype(), device=self.tensor.data.device
        )
        if integer_mode:
            image = ip.place_patches_integer(image, positions, patches, op="add")
        else:
            image = self.place_patches_function(
                image, positions, patches, op="add", pad=pad_for_shift
            )
        return image
    
    def get_object_in_roi(self):
        bbox = self.roi_bbox.get_bbox_with_top_left_origin()
        return self.data[:, int(bbox.sy):int(bbox.ey), int(bbox.sx):int(bbox.ex)]

    # @timer()
    # def constrain_smoothness(self) -> None:
    #     """
    #     Smooth the magnitude of the object.
    #     """
    #     if self.options.smoothness_constraint.alpha <= 0:
    #         return
    #     alpha = self.options.smoothness_constraint.alpha
    #     if alpha > 1.0 / 8:
    #         logger.warning(f"Alpha = {alpha} in smoothness constraint should be less than 1/8.")
    #     psf = torch.ones(3, 3, device=self.tensor.data.device) * alpha
    #     psf[2, 2] = 1 - 8 * alpha

    #     data = self.data
    #     for i_slice in range(self.n_slices):
    #         slc0 = data[i_slice]
    #         slc = ip.convolve2d(slc0, psf, "same")
    #         data[i_slice] = slc
    #     self.set_data(data)

    @timer()
    def constrain_smoothness(self) -> None:
        """
        Smooth the magnitude of the object.
        """
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        data = self.data
        

         
         
         
         
         
         
        # from torchvision.transforms.functional import gaussian_blur
        # Xabs = gaussian_blur( torch.abs(data), kernel_size=(3, 3), sigma=(2.0, 2.0)) 
        # data = Xabs * torch.exp(1j * torch.angle(data))
         
        # from torchvision.transforms.functional import gaussian_blur
        # Xre = gaussian_blur( torch.real(data), kernel_size=(3, 3), sigma=(2.0, 2.0)) 
        # Xim = gaussian_blur( torch.imag(data), kernel_size=(3, 3), sigma=(2.0, 2.0))
        # data = Xre + 1j * Xim
         

         
         
        # import torch.nn.functional as F
        
        # def median_filter_2d(input_tensor, kernel_size):
        #     """Custom 2D median filter implementation using PyTorch operations."""
        #     # Store original shape
        #     original_shape = input_tensor.shape
            
        #     # Add batch and channel dimensions if needed
        #     if input_tensor.dim() == 2:
        #         input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        #     elif input_tensor.dim() == 3:
        #         input_tensor = input_tensor.unsqueeze(0)
            
        #     # Calculate padding
        #     pad = kernel_size // 2
            
        #     # Pad the input tensor
        #     padded = F.pad(input_tensor, (pad, pad, pad, pad), mode='reflect')
            
        #     # Unfold to get patches
        #     patches = F.unfold(padded, kernel_size=kernel_size, stride=1)
            
        #     # Reshape patches for median calculation
        #     # patches shape: [batch_size, channels * kernel_size^2, num_patches]
        #     batch_size, channels_times_kernel_sq, num_patches = patches.shape
        #     channels = input_tensor.shape[1]
        #     patches = patches.view(batch_size, channels, kernel_size * kernel_size, num_patches)
            
        #     # Calculate median along the kernel dimension
        #     median_patches = torch.median(patches, dim=2)[0]
            
        #     # Reshape back to spatial dimensions
        #     h, w = input_tensor.shape[-2:]
        #     median_filtered = median_patches.view(batch_size, channels, h, w)
            
        #     # Remove added dimensions and return to original shape
        #     result = median_filtered.squeeze()
        #     if len(original_shape) == 2:
        #         return result
        #     else:
        #         return result.view(original_shape)
        
        # # kernel_size = 9  
        # # Xre = median_filter_2d(torch.real(data[0,...]), kernel_size=kernel_size)
        # # Xim = median_filter_2d(torch.imag(data[0,...]), kernel_size=kernel_size)
        # # data[0,...] = Xre + 1j * Xim
        
        # kernel_size = 3  
        # Xabs = median_filter_2d(torch.abs(data[0,...]), kernel_size=kernel_size)
        # data[0,...] = Xabs * torch.exp(1j * torch.angle(data[0,...]))
         
         
         
         
         
         
         
         
        
        # if self.options.smoothness_constraint.alpha <= 0:
        #     return
        # alpha = self.options.smoothness_constraint.alpha
        # if alpha > 1.0 / 8:
        #     logger.warning(f"Alpha = {alpha} in smoothness constraint should be less than 1/8.")
        # psf = torch.ones(3, 3, device=self.tensor.data.device) * alpha
        # psf[2, 2] = 1 - 8 * alpha

        # data = self.data
        # for i_slice in range(self.n_slices):
        #     slc0 = data[i_slice]
        #     slc = ip.convolve2d(slc0, psf, "same")
        #     data[i_slice] = slc
         
         
         
         
         
         
         
         
         
         
         
         
         
         
        phs_data = torch.angle( data )
        abs_data = torch.abs( data )
        
        limHI = 5.00
        limLO = 0.01
        maskHI = ( abs_data > limHI ) 
        maskLO = ( abs_data < limLO ) 
        maskOK = torch.logical_not( maskHI ) * torch.logical_not( maskLO )
        #abs_data  = ( limLO * maskLO + limHI * maskHI + maskOK * abs_data ) 
        abs_data  = ( limLO * maskLO + 0.5 * abs_data * maskHI + maskOK * abs_data ) 
        
        limHI = +0.5 * torch.pi
        limLO = -0.5 * torch.pi
        maskHI = ( abs_data > limHI ) 
        maskLO = ( abs_data < limLO ) 
        maskOK = torch.logical_not( maskHI ) * torch.logical_not( maskLO )
        #phs_data  = ( limLO * maskLO + limHI * maskHI + maskOK * phs_data ) 
        phs_data  = ( limLO * maskLO + 0.5 * phs_data * maskHI + maskOK * phs_data ) 
        
        data = abs_data * torch.exp( 1j * phs_data )

            
            
            
        self.set_data(data)




        
        # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        # rows, cols = self.data[0,...].shape[-2:]
        # center_r = int( rows // 2 - 0 * rows / 100 ) 
        # center_c = int( cols // 2 - 0 * cols / 100 ) 
        # radius_r = ( rows * 0.25 )
        # radius_c = ( cols * 0.20 )
        # y, x = torch.meshgrid( torch.arange(rows), torch.arange(cols), indexing='ij' )
        # dist = ((y - center_r)**2 / radius_r**2) + ((x - center_c)**2 / radius_c**2)
        # fixed_support = dist <= 1
  

        # # y_coords, x_coords = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing='ij')

        # # # Translate coordinates to ellipse center
        # # x_translated = x_coords - center_x
        # # y_translated = y_coords - center_y

        # # # Convert angle to radians
        # # angle_radians = math.radians(1)

        # # # Rotate coordinates
        # # x_rotated = x_translated * math.cos(angle_radians) + y_translated * math.sin(angle_radians)
        # # y_rotated = -x_translated * math.sin(angle_radians) + y_translated * math.cos(angle_radians)

        # # # Apply ellipse equation
        # # ellipse_equation_result = (x_rotated / semi_major_axis)**2 + (y_rotated / semi_minor_axis)**2

        # # mask = (ellipse_equation_result <= 1).float()
    
  
  
        
        # # fixed_support = torch.zeros(self.data.shape[-2:], dtype = torch.float32)
        # # rows, cols = self.data.shape[-2:]
        # # center_r = int( rows // 2 - 1 * rows / 100 ) 
        # # center_c = int( cols // 2 - 1 * cols / 100 ) 
        # # len_r = int( rows * 0.20 )
        # # len_c = int( cols * 0.15 )
        # # fixed_support[ center_r - len_r : center_r + len_r, center_c - len_c : center_c + len_c ] = 1
 
 
 
 
        # w = 0.3
        # data = w * self.data * fixed_support[None,...] + (1-w) * self.data
        
        # # w = 0.5
        # # data = torch.abs( data ) * torch.exp(1j * ( w * torch.angle( data ) + (1-w) *  torch.angle( data ) * fixed_support[None,...] ))
  
        # #data[torch.abs(data)==0] = 0.9

        # self.set_data(data)
        
        # '''
        

        # import numpy as np
        # import matplotlib.pyplot as plt
        
        # plt.figure(); plt.imshow(np.abs( self.data[0,:,:].cpu().numpy())); 
        # plt.savefig('/home/beams/ATRIPATH/Desktop/beamtime_092025_9id/fixed_support.png', dpi=600)
        # plt.close('all')
        
        # plt.figure(); plt.imshow(np.abs( ( self.data[0,:,:] * fixed_support ).cpu().numpy())); 
        # plt.savefig('/home/beams/ATRIPATH/Desktop/beamtime_092025_9id/object_fixed_support.png', dpi=600)
        # plt.close('all')
        
        # '''
        
        
        
        
        
        # # if self.options.l2_norm_constraint.weight <= 0:
        # #     return
        # # data = self.data
        # # l2_grad = data * 2 / data.numel()
        # # data = data - self.options.l2_norm_constraint.weight * l2_grad
        # # self.set_data(data)
        # # logger.debug("L2 norm constraint applied to object.")



        
        # from torchvision.transforms.functional import gaussian_blur
        
        # ks = int( self.options.l1_norm_constraint.weight )
        # sg = float( self.options.l1_norm_constraint.weight - 1)
        
        # # data_blur = gaussian_blur( torch.abs(data), kernel_size=(ks, ks), sigma=(sg, sg)) 
        # # w = 0.5
        # # data = w * (data_blur * torch.exp(1j * torch.angle(data))) + (1-w) * data
        
        # from torchvision.transforms.functional import gaussian_blur
        # Xre = gaussian_blur( torch.real(data), kernel_size=(ks, ks), sigma=(sg, sg)) 
        # Xim = gaussian_blur( torch.imag(data), kernel_size=(ks, ks), sigma=(sg, sg))
        # data_blur = Xre + 1j * Xim

        # w = 0.7
        # data = w * data_blur + (1-w) * data
        
        
        
        # # if self.options.l1_norm_constraint.weight <= 0:
        # #     return
        # # l1_grad = torch.sgn(data) / data.numel()
        # # data = data - self.options.l1_norm_constraint.weight * l1_grad
        
        























    # @timer()
    # def constrain_total_variation(self) -> None:
    #     if self.options.total_variation.weight <= 0:
    #         return
    #     data = self.data
    #     for i_slice in range(self.n_slices):
    #         data[i_slice] = ip.total_variation_2d_chambolle(
    #             data[i_slice], lmbda=self.options.total_variation.weight, niter=2
    #         )
    #     self.set_data(data)
    
    @timer()
    def constrain_total_variation(self) -> None:
        
        # if self.options.total_variation.weight < 0:            # if self.options.total_variation.weight <= 0:
        #     return
        
    
        data = self.data
        
         
        # from torchvision.transforms.functional import gaussian_blur
        # Xre = gaussian_blur( torch.real(data), kernel_size=(3, 3), sigma=(2.0, 2.0)) 
        # Xim = gaussian_blur( torch.imag(data), kernel_size=(3, 3), sigma=(2.0, 2.0))
        # data = Xre + 1j * Xim
        
        '''
        import numpy as np
        import matplotlib.pyplot as plt
        plt.figure(); plt.imshow(np.abs( data[0,...].cpu().numpy())); 
        plt.savefig('/home/beams/ATRIPATH/Desktop/beamtime_092025_9id/object_fixed_support.png')

        '''
        

        if self.options.total_variation.weight > 0:
            for i_slice in range(self.n_slices):
                data[i_slice] = ip.total_variation_2d_chambolle(
                    data[i_slice], lmbda=self.options.total_variation.weight, niter=2
                )
            

        self.set_data(data)

    @timer()
    def remove_grid_artifacts(self):
        data = self.data
        for i_slice in range(self.n_slices):
            if self.options.remove_grid_artifacts.component == enums.MagPhaseComponents.PHASE:
                slice_data = torch.angle(data[i_slice])
            elif self.options.remove_grid_artifacts.component == enums.MagPhaseComponents.MAGNITUDE:
                slice_data = torch.abs(data[i_slice])
            elif self.options.remove_grid_artifacts.component == enums.MagPhaseComponents.BOTH:
                slice_data = data[i_slice]
            else:
                raise ValueError(f"Invalid value for `component`: {self.options.remove_grid_artifacts.component}")
            slice_data = ip.remove_grid_artifacts(
                slice_data,
                pixel_size_m=self.pixel_size_m,
                period_x_m=self.options.remove_grid_artifacts.period_x_m,
                period_y_m=self.options.remove_grid_artifacts.period_y_m,
                window_size=self.options.remove_grid_artifacts.window_size,
                direction=self.options.remove_grid_artifacts.direction,
            )
            if self.options.remove_grid_artifacts.component == enums.MagPhaseComponents.PHASE:
                data[i_slice] = data[i_slice].abs() * torch.exp(1j * slice_data)
            elif self.options.remove_grid_artifacts.component == enums.MagPhaseComponents.MAGNITUDE:
                data[i_slice] = torch.sgn(data[i_slice]) * torch.abs(slice_data)
            elif self.options.remove_grid_artifacts.component == enums.MagPhaseComponents.BOTH:
                data[i_slice] = slice_data
        self.set_data(data)

    @timer()
    def regularize_multislice(self):
        """ 
        Regularize multislice by applying a low-pass transfer function to the
        3D Fourier space of the magnitude and phase (unwrapped) of all slices.

        Adapted from fold_slice (regulation_multilayers.m).
        """
        if not self.is_multislice or self.options.multislice_regularization.weight <= 0:
            return
        if self.preconditioner is None:
            raise ValueError("Regularization requires a preconditioner.")

        # TODO: use CPU if GPU memory usage is too large.
        fourier_coords = []
        for s in (self.n_slices, self.lateral_shape[0], self.lateral_shape[1]):
            u = torch.fft.fftfreq(s)
            fourier_coords.append(u)
        fourier_coords = torch.meshgrid(*fourier_coords, indexing="ij")
        # Calculate force of regularization based on the idea that DoF = resolution^2/lambda
        w = 1 - torch.atan(
            (
                self.options.multislice_regularization.weight
                * torch.abs(fourier_coords[0])
                / torch.sqrt(fourier_coords[1] ** 2 + fourier_coords[2] ** 2 + 1e-3)
            )
            ** 2
        ) / (torch.pi / 2)
        relax = 1
        alpha = 1
        # Low-pass transfer function.
        w_a = w * torch.exp(-alpha * (fourier_coords[1] ** 2 + fourier_coords[2] ** 2))
        obj = self.data

        # Find correction for amplitude.
        aobj = torch.abs(obj)
        fobj = torch.fft.fftn(aobj)
        fobj = fobj * w_a
        aobj_upd = torch.fft.ifftn(fobj)
        # Push towards zero.
        aobj_upd = 1 + 0.9 * (aobj_upd - 1)

        # Find correction for phase.
        w_phase = torch.clip(10 * (self.preconditioner / self.preconditioner.max()), max=1)

        if self.options.multislice_regularization.unwrap_phase:
            pobj = [
                ip.unwrap_phase_2d(
                    obj[i_slice],
                    weight_map=w_phase,
                    fourier_shift_step=0.5,
                    image_grad_method=self.options.multislice_regularization.unwrap_image_grad_method,
                    image_integration_method=self.options.multislice_regularization.unwrap_image_integration_method,
                )
                for i_slice in range(self.n_slices)
            ]
            pobj = torch.stack(pobj, dim=0)
        else:
            pobj = torch.angle(obj)
        fobj = torch.fft.fftn(pobj)
        fobj = fobj * w_a
        pobj_upd = torch.fft.ifftn(fobj)

        aobj_upd = torch.real(aobj_upd) - aobj
        pobj_upd = w_phase * (torch.real(pobj_upd) - pobj)
        corr = (1 + relax * aobj_upd) * torch.exp(1j * relax * pobj_upd)
        obj = obj * corr
        self.set_data(obj)
        
    def remove_object_probe_ambiguity(
        self, 
        probe: "ds.probe.Probe", 
        *, 
        update_probe_in_place: bool = True
    ) -> Optional[Tensor]:
        """
        Remove the object-probe ambiguity by scaling the object by its norm,
        and adjusting the probe power accordingly.
        
        Parameters
        ----------
        probe : ds.probe.Probe
            The probe to adjust.
        update_probe_in_place : bool, optional
            Whether to update the probe in place. If False, the tensor of the updated probe is returned.
        """
        bbox = self.roi_bbox.get_bbox_with_top_left_origin()
        roi_slicer = bbox.get_slicer()
        w = self.preconditioner[*roi_slicer]
        w = w / pmath.mnorm(w, dim=(-2, -1))
        
        # Get the norm of the object within the ROI for each slice.
        obj_data = self.data
        obj_norm = torch.sqrt(torch.mean(torch.abs(obj_data[..., *roi_slicer]) ** 2 * w, dim=(-2, -1)))
        
        # Scale the object such that the mean transmission is 1.
        obj_data = obj_data / obj_norm[:, None, None]
        
        # Adjust the probe power accordingly.
        probe_data = probe.data
        probe_data = probe_data * torch.prod(obj_norm)
        
        self.set_data(obj_data)
        if update_probe_in_place:
            probe.set_data(probe_data)
        else:
            return probe_data
        
        
    def calculate_illumination_map(
        self, 
        probe: "ds.probe.Probe",
        probe_positions: "ds.probe_positions.ProbePositions",
        use_all_modes: bool = False
    ) -> Tensor:
        """Calculate the illumination map by overlaying the probe intensity
        at all positions on a zero array. The map is used to calculate 
        the preconditioner.
        
        Parameters
        ----------
        probe : ds.probe.Probe
            The probe to use for the illumination map.
        probe_positions : ds.probe_positions.ProbePositions
            The positions of the probe.
        use_all_modes : bool, optional
            Whether to use all modes of the probe.

        Returns
        -------
        Tensor
            The illumination map of the object.
        """
        positions_all = probe_positions.tensor
        # Shape of probe:        (n_probe_modes, h, w)
        object_ = self.get_slice(0)

        if use_all_modes:
            probe_int = probe.get_all_mode_intensity(opr_mode=0)[None, :, :]
        else:
            probe_int = probe.get_mode_and_opr_mode(mode=0, opr_mode=0)[None, ...].abs() ** 2

        # Stitch probes of all positions on the object buffer
        # TODO: allow setting chunk size externally
        probe_sq_map = chunked_processing(
            func=ip.place_patches_integer,
            common_kwargs={"op": "add"},
            chunkable_kwargs={
                "positions": positions_all.round().int() + self.pos_origin_coords,
            },
            iterated_kwargs={
                "image": torch.zeros_like(object_.real).type(torch.get_default_dtype())
            },
            replicated_kwargs={
                "patches": probe_int,
            },
            chunk_size=64,
        )
        return probe_sq_map

    def update_preconditioner(
        self,
        probe: "ds.probe.Probe",
        probe_positions: "ds.probe_positions.ProbePositions",
        patterns: Tensor = None,
        use_all_modes: bool = False,
    ) -> None:
        """Update the preconditioner. This function reproduces the behavior
        of PtychoShelves in `ptycho_solver`: it averages the new illumination
        map and the old preconditioner to reduce oscillations.
        
        Parameters
        ----------
        probe : ds.probe.Probe
            The probe to use for the illumination map.
        probe_positions : ds.probe_positions.ProbePositions
            The positions of the probe.
        patterns : Tensor, optional
            A (n_scan_points, h, w) tensor giving the diffraction patterns. Only needed
            if the preconditioner does not exist and needs to be initialized.
        use_all_modes : bool, optional
            Whether to use the sum of all probe modes' intensities for the illumination map.
        """
        if self.preconditioner is None:
            self.initialize_preconditioner(probe, probe_positions, patterns)
        illum_map = self.calculate_illumination_map(probe, probe_positions, use_all_modes=use_all_modes)
        self.preconditioner = (self.preconditioner + illum_map) / 2
    
    def initialize_preconditioner(
        self, 
        probe: "ds.probe.Probe", 
        probe_positions: "ds.probe_positions.ProbePositions",
        patterns: Tensor,
    ) -> None:
        """Initialize the preconditioner. This function reproduces the behavior
        of PtychoShelves in `init_solver` and `load_from_p`: the probe is first
        renormalized before being used to calculate the illumination map.
        Diffraction patterns are needed to calculate the renormalization factor.
        
        Parameters
        ----------
        probe : ds.probe.Probe
            The probe to use for the illumination map.
        probe_positions : ds.probe_positions.ProbePositions
            The positions of the probe.
        patterns : ds.patterns.Patterns
            A (n_scan_points, h, w) tensor giving the diffraction patterns.
        """
        probe_data = probe.data
        probe_renormalization_factor = get_probe_renormalization_factor(patterns)
        probe_data = probe_data / (math.sqrt(probe.shape[-1] * probe.shape[-2]) * 2 * probe_renormalization_factor)
        probe_temp = ds.probe.Probe(data=probe_data, options=copy.deepcopy(probe.options))
        self.preconditioner = self.calculate_illumination_map(probe_temp, probe_positions, use_all_modes=False)


class DIPObject(Object):
    
    options: "api.options.ad_ptychography.AutodiffPtychographyObjectOptions"
    
    def __init__(
        self,
        name: str = "object",
        options: "api.options.ad_ptychography.AutodiffPtychographyObjectOptions" = None,
        *args,
        **kwargs
    ) -> None:
        """Deep image prior object.

        Parameters
        ----------
        name : str, optional
            The name of the object.
        options : api.options.base.ObjectOptions, optional
            The options for the object.
        """
        super().__init__(name=name, options=options, *args, **kwargs)
        self.model = None
        self.dip_output_magnitude = None
        self.dip_output_phase = None
        
        self.build_model()
        self.build_dip_optimizer()
        
        # `self.tensor` is used to hold the object generated by the DIP model and
        # is not trainable.
        self.tensor.requires_grad_(False)
        
        self.initial_data = None
        if self.options.experimental.deep_image_prior_options.residual_generation:
            self.initial_data = self.data.clone()   
        
    def build_model(self):
        if not self.options.experimental.deep_image_prior_options.enabled:
            return
        model_class = maps.get_nn_model_by_enum(self.options.experimental.deep_image_prior_options.model)
        self.model = model_class(
            **self.options.experimental.deep_image_prior_options.model_params
        )
        
    def build_dip_optimizer(self):
        if self.optimizable and self.optimizer_class is None:
            raise ValueError(
                "Parameter {} is optimizable but no optimizer is specified.".format(self.name)
            )
        if self.optimizable:
            self.optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_params)
        
    def generate(self):
        raise NotImplementedError


class DIPPlanarObject(DIPObject, PlanarObject):
    def __init__(
        self,
        name: str = "object",
        options: "api.options.base.ObjectOptions" = None,
        *args,
        **kwargs
    ) -> None:
        """Deep image prior planar object.

        Parameters
        ----------
        name : str, optional
            The name of the object.
        options : api.options.base.ObjectOptions, optional
            The options for the object.
        """
        super().__init__(name=name, options=options, data_as_parameter=False, *args, **kwargs)
        
        nn_input = self.get_nn_input()
        self.register_buffer("nn_input", nn_input)
        
    def get_nn_input(self):
        z = torch.rand(
            [self.n_slices, self.options.experimental.deep_image_prior_options.net_input_channels, *self.lateral_shape], 
        ) * 0.1
        return z

    def generate(self) -> Tensor:
        """Generate the object using the deep image prior model, and set the
        generated object to self.data.
        
        Returns
        -------
        Tensor
            The generated object.
        """
        if self.model is None:
            raise ValueError("Model is not built.")
        o = self.model(self.nn_input)
        
        o, mag, phase = self.process_net_output(o)
        
        with torch.no_grad():
            self.dip_output_magnitude = mag.clone()
            self.dip_output_phase = phase.clone()
        
        if self.options.experimental.deep_image_prior_options.residual_generation:
            init_data = torch.stack([self.initial_data.real, self.initial_data.imag], dim=-1)
            o = o + init_data
        self.tensor.data = o
        return self.data

    def process_net_output(self, o):
        """Apply constraints to the DIP network's output and transform the tensor.
        For the magnitude, it is clipped at 0 and then added by 1.0.

        Parameters
        ----------
        o : Tensor | tuple[Tensor, Tensor]
            The output of the DIP network. It should either be a [n_slices, 2, h, w] 
            tensor with the channels giving the magnitude and phase of the object,
            or a tuple of two [n_slices, 1, h, w] tensors giving the magnitude and phase
            of the object.
            
        Returns
        -------
        Tensor
            A [n_slices, h, w, 2] tensor representing the real and imaginary parts 
            of the object.
        Tensor
            The magnitude of the object.
        Tensor
            The phase of the object.
        """
        if isinstance(o, (list, tuple)):
            mag, phase = o
            mag = mag[:, 0]
            phase = phase[:, 0]
        else:
            mag = o[:, 0]
            phase = o[:, 1]
            
        expected_phase_shape = (self.n_slices, *self.lateral_shape)
        if tuple(phase.shape) != expected_phase_shape:
            logger.warning(
                "Shape of phase output by NN is expected to be "
                f"(n_slices, h, w) = {expected_phase_shape} but got {phase.shape}. "
                "Padding/cropping generated object to match expected shape."
            )
            mag = ip.central_crop_or_pad(mag, expected_phase_shape[1:])
            phase = ip.central_crop_or_pad(phase, expected_phase_shape[1:])
        
        if self.options.experimental.deep_image_prior_options.constrain_object_outside_network:
            mag = torch.sigmoid(mag)
        
        o_complex = mag * torch.exp(1j * phase)
        o = torch.stack([o_complex.real, o_complex.imag], dim=-1)
        return o, mag, phase
