import os, glob, time
import h5py
import hdf5plugin  # for reading raw hdf5 files
import numpy as np
import scipy.ndimage
from scipy.interpolate import interp1d
from tifffile import imwrite
from ptychi.image_proc import unwrap_phase_2d
import ptychi.api as api
from ptychi.utils import (
    add_additional_opr_probe_modes_to_probe,
    to_tensor,
    orthogonalize_initial_probe,
    get_suggested_object_size,
    get_default_complex_dtype,
)
from ptychi.maths import compose_2x2_affine_transform_matrix
import matplotlib.pyplot as plt

# from matplotlib.patches import Rectangle
from ptychi.pear_utils import make_fzp_probe, resize_complex_array, find_matching_recon, crop_pad, verbose_print
from ptychi.pear_aps_data_loaders import (
    load_data_2xfm,
    load_data_12idc,
    load_data_bnp,
    load_data_lynx,
    load_data_velo
)
import sys
import torch
import torch.distributed as dist
import json
import shutil

# Global variable for print mode
print_mode = 'debug'

def save_reconstructions(task, recon_path, iter, params):
    # Only save from rank 0 to avoid file locking issues in distributed training
    if task.rank != 0:
        return
    
    if params.get("beam_source", "xray") == "electron":
        pixel_size = task.object_options.pixel_size_m * 1e9
        pixel_unit = "nm"
    else:
        pixel_size = task.object_options.pixel_size_m * 1e6
        pixel_unit = "um"

    # Object
    recon_object = task.get_data_to_cpu("object", as_numpy=True)
    if params.get("save_full_object", False):
        recon_object_roi = torch.from_numpy(recon_object)
    else:
        recon_object_roi = task.object.get_object_in_roi().cpu().detach()

    if recon_object_roi.shape[0] > 1:  # multislice recon
        # object_ph_stack = [normalize_by_bit_depth(unwrap_phase_2d(slice.cuda(),
        #                                                          image_grad_method='fourier_differentiation',
        #                                                          image_integration_method='fourier').cpu(), '16')
        #                   for slice in recon_object_roi]
        # Unwrap phase for each slice
        unwrapped_phases = [
            unwrap_phase_2d(
                slice.cuda(),
                image_grad_method="fourier_differentiation",
                image_integration_method="fourier",
            )
            .cpu()
            .numpy()
            for slice in recon_object_roi
        ]

        # Find global min and max for normalization
        global_min = min(phase.min() for phase in unwrapped_phases)
        global_max = max(phase.max() for phase in unwrapped_phases)

        # Check if the range is too small, which can cause contrast issues
        if global_max - global_min < 1e-6:
            print(
                "Warning: Very small global range detected in phase values. Using per-slice normalization."
            )
            object_ph_stack = [normalize_by_bit_depth(phase, "16") for phase in unwrapped_phases]
        else:
            # Apply robust normalization with clipping to improve contrast
            # Calculate percentiles for robust scaling (removes extreme outliers)
            all_phases = np.concatenate([phase.flatten() for phase in unwrapped_phases])
            p_low, p_high = np.percentile(all_phases, [1, 99])

            # print(f"Global phase range: {global_min:.4f} to {global_max:.4f}")
            # print(f"Robust phase range (1-99 percentile): {p_low:.4f} to {p_high:.4f}")

            # Normalize all slices using robust global range with clipping
            object_ph_stack = []
            for phase in unwrapped_phases:
                # Clip to robust range
                phase_clipped = np.clip(phase, p_low, p_high)
                # Normalize to 16-bit range
                normalized = (phase_clipped - p_low) / (p_high - p_low) * 65535
                object_ph_stack.append(np.uint16(normalized))

        imwrite(
            f"{recon_path}/object_ph_layers/object_ph_layers_Niter{iter}.tiff",
            np.array(object_ph_stack),
            photometric="minisblack",
            resolution=(1 / pixel_size, 1 / pixel_size),
            metadata={"unit": pixel_unit, "pixel_size": pixel_size},
            imagej=True,
        )

        object_ph_sum = normalize_by_bit_depth(
            unwrap_phase_2d(
                torch.prod(recon_object_roi, dim=0).cuda(),
                image_grad_method="fourier_differentiation",
                image_integration_method="fourier",
            ).cpu(),
            "16",
        )
        imwrite(
            f"{recon_path}/object_ph_total/object_ph_total_Niter{iter}.tiff",
            np.array(object_ph_sum),
            photometric="minisblack",
            resolution=(1 / pixel_size, 1 / pixel_size),
            metadata={"unit": pixel_unit, "pixel_size": pixel_size},
            imagej=True,
        )

        object_mag_stack = [
            normalize_by_bit_depth(np.abs(slice), "16") for slice in recon_object_roi
        ]
        imwrite(
            f"{recon_path}/object_mag_layers/object_mag_layers_Niter{iter}.tiff",
            np.array(object_mag_stack),
            photometric="minisblack",
            resolution=(1 / pixel_size, 1 / pixel_size),
            metadata={"unit": pixel_unit, "pixel_size": pixel_size},
            imagej=True,
        )

        object_mag_sum = normalize_by_bit_depth(
            np.abs(torch.prod(recon_object_roi, dim=0)).cpu(), "16"
        )
        imwrite(
            f"{recon_path}/object_mag_total/object_mag_total_Niter{iter}.tiff",
            np.array(object_mag_sum),
            photometric="minisblack",
            resolution=(1 / pixel_size, 1 / pixel_size),
            metadata={"unit": pixel_unit, "pixel_size": pixel_size},
            imagej=True,
        )

    else:
        # imwrite(f'{recon_path}/object_ph/object_ph_roi_Niter{iter}.tiff', normalize_by_bit_depth(np.angle(recon_object_roi[0,]), '16'))
        object_ph_unwrapped = unwrap_phase_2d(
            recon_object_roi[0,].cuda(),
            image_grad_method="fourier_differentiation",
            image_integration_method="fourier",
        )
        # object_ph_unwrapped = np.angle(recon_object_roi[0,].cuda())
        imwrite(
            f"{recon_path}/object_ph/object_ph_Niter{iter}.tiff",
            normalize_by_bit_depth(object_ph_unwrapped.cpu(), "16"),
            photometric="minisblack",
            resolution=(1 / pixel_size, 1 / pixel_size),
            metadata={"unit": pixel_unit, "pixel_size": pixel_size},
            imagej=True,
        )
        imwrite(
            f"{recon_path}/object_mag/object_mag_Niter{iter}.tiff",
            normalize_by_bit_depth(np.abs(recon_object_roi[0,]), "16"),
            photometric="minisblack",
            resolution=(1 / pixel_size, 1 / pixel_size),
            metadata={"unit": pixel_unit, "pixel_size": pixel_size},
            imagej=True,
        )

    # Calculate the phase
    #  recon_object_roi_ph = unwrap_phase_2d(recon_object_roi[0,].cuda(), image_grad_method='fourier_differentiation', image_integration_method='fourier')
    # imwrite(f'{recon_path}/object_ph/object_ph_unwrap_roi_Niter{iter}.tiff', normalize_by_bit_depth(recon_object_roi_ph.cpu(), '16'))

    # Probe
    recon_probe = task.get_data_to_cpu("probe", as_numpy=True)
    probe_mag = np.hstack(np.abs(recon_probe[0,]))
    # plt.imsave(f'{recon_path}/probe_mag/probe_mag_Niter{iter}.png', probe_mag, cmap='plasma')

    norm = plt.Normalize(vmin=probe_mag.min(), vmax=probe_mag.max())
    cmap = plt.cm.plasma
    colored_probe_mag = cmap(norm(probe_mag))  # This creates an RGBA array
    colored_probe_mag = (colored_probe_mag[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB uint8

    # Save with ImageJ-compatible resolution information
    imwrite(
        f"{recon_path}/probe_mag/probe_mag_Niter{iter}.tiff",
        colored_probe_mag,
        photometric="rgb",
        resolution=(1 / pixel_size, 1 / pixel_size),
        metadata={"unit": pixel_unit, "pixel_size": pixel_size},
        imagej=True,
    )

    # Save probe propagation in multislice reconstruction
    if recon_object_roi.shape[0] > 1:
        from ptychi.pear_utils import near_field_evolution

        # Get the primary probe mode
        # Extract the primary probe mode based on dimensionality
        probe = (
            recon_probe[0, 0]
            if recon_probe.ndim == 4
            else (recon_probe[0] if recon_probe.ndim == 3 else recon_probe)
        )

        # Check if slice_spacings attribute exists in the correct location
        # Get slice spacings directly from object_options
        slice_spacings = task.object_options.slice_spacings_m
        n_layers = len(slice_spacings) + 1

        # Initialize array to store propagated probes
        probe_propagation = np.zeros((n_layers, probe.shape[0], probe.shape[1]), dtype=np.complex64)
        probe_propagation[0] = probe  # First layer is the original probe

        # Physical size of the array (extent)
        extent = probe.shape[0] * task.object_options.pixel_size_m

        # Get wavelength
        if hasattr(task.reconstructor.parameter_group, "wavelength"):
            wavelength = task.reconstructor.parameter_group.wavelength.cpu().numpy()
        else:
            wavelength = task.data_options.wavelength_m

        # Propagate probe through each layer
        for i in range(1, n_layers):
            # Propagate from previous layer to current layer
            z_distance = slice_spacings[i - 1]
            try:
                u_1, _, _, _ = near_field_evolution(
                    probe_propagation[i - 1], z_distance, wavelength, extent, use_ASM_only=True
                )
                probe_propagation[i] = u_1
            except Exception as e:
                print(f"Warning: Error during probe propagation at layer {i}: {str(e)}")
                # Copy previous layer as fallback
                probe_propagation[i] = probe_propagation[i - 1]

        # Create colored versions of the propagated probes
        colored_probes_mag_stack = []
        for i in range(n_layers):
            probe_mag = np.abs(probe_propagation[i])
            # Normalize the probe magnitude
            norm = plt.Normalize(vmin=probe_mag.min(), vmax=probe_mag.max())
            # Apply colormap
            colored_probe = cmap(norm(probe_mag))  # This creates an RGBA array
            # Convert to RGB uint8
            colored_probe_rgb = (colored_probe[:, :, :3] * 255).astype(np.uint8)
            colored_probes_mag_stack.append(colored_probe_rgb)

        # Save as tiff stack
        imwrite(
            f"{recon_path}/probe_propagation_mag/probe_propagation_mag_Niter{iter}.tiff",
            np.array(colored_probes_mag_stack),
            photometric="rgb",
            resolution=(1 / pixel_size, 1 / pixel_size),
            metadata={"unit": pixel_unit, "pixel_size": pixel_size},
            imagej=True,
        )

    # # Create figure and axis
    # fig, ax = plt.subplots()

    # # Display the image
    # im = ax.imshow(probe_mag, cmap='plasma')

    # # Add scale bar
    # bar_length_pixels = 1e-6/task.object_options.pixel_size_m  # Length of scale bar in pixels
    # bar_width_pixels = N_probe/30    # Width of scale bar in pixels
    # bar_position_x = N_probe/20     # from left edge
    # bar_position_y = N_probe*0.9  # from bottom

    # # Create and add the scale bar
    # scale_bar = Rectangle((bar_position_x, bar_position_y),
    #                     bar_length_pixels, bar_width_pixels,
    #                     fc='white', ec='none')
    # ax.add_patch(scale_bar)

    # # Optional: Add text above scale bar
    # plt.text(bar_position_x + bar_length_pixels/2,
    #         bar_position_y - 3,
    #         '1 um',
    #         color='white',
    #         ha='center',
    #         va='bottom',
    #         fontsize=5)

    # # Remove axes
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_frame_on(False)

    # # Save the figure
    # plt.savefig(f'{recon_path}/probe_mag/probe_mag_Niter{iter}.png',
    #             dpi=500,
    #             bbox_inches='tight',
    #             pad_inches=0)
    # plt.close()

    # plt.imsave(f'{recon_path}/probe_mag/probe_mag_Niter{iter}.tiff', normalize_by_bit_depth(probe_mag, '16'), cmap='plasma')
    # imwrite(f'{recon_path}/probe_mag/pxrobe_mag_Niter{iter}.tiff', normalize_by_bit_depth(probe_mag, '16'))
    # Save probe at each scan position
    opr_mode_weights = (
        task.reconstructor.parameter_group.opr_mode_weights.data.cpu().detach().numpy()
    )
    if recon_probe.shape[0] > 1 and params.get("save_probe_at_each_scan_position", False):
        probes = task.reconstructor.parameter_group.probe.get_unique_probes(
            task.reconstructor.parameter_group.opr_mode_weights.data, mode_to_apply=0
        )
        probes = probes[:, 0, :, :].cpu().detach().numpy()  # only keep the primary mode

        # Create a colored version of each probe magnitude
        colored_probes_mag_stack = []
        for i in range(min(probes.shape[0], 500)):
            probe_mag = np.abs(probes[i])
            # Normalize the probe magnitude
            norm = plt.Normalize(vmin=probe_mag.min(), vmax=probe_mag.max())
            # Apply colormap
            colored_probe = cmap(norm(probe_mag))  # This creates an RGBA array
            # Convert to RGB uint8
            colored_probe_rgb = (colored_probe[:, :, :3] * 255).astype(np.uint8)
            colored_probes_mag_stack.append(colored_probe_rgb)

        # Save as tiff stack
        imwrite(
            f"{recon_path}/probe_mag_opr/probes_mag_opr_Niter{iter}.tiff",
            np.array(colored_probes_mag_stack),
            photometric="rgb",
            resolution=(1 / pixel_size, 1 / pixel_size),
            metadata={"unit": pixel_unit, "pixel_size": pixel_size},
            imagej=True,
        )

    # Save scan positions
    scan_positions = task.get_data_to_cpu("probe_positions", as_numpy=True)
    if params["position_correction"]:
        if params.get("save_plots", True):
            plt.figure()
            plt.scatter(-init_positions_x, init_positions_y, s=1, edgecolors="blue")
            plt.scatter(
                -scan_positions[:, 1] * pixel_size,
                scan_positions[:, 0] * pixel_size,
                s=10,
                edgecolors="red",
                facecolors="none",
            )
            # Calculate average position differences
            x_diff = np.mean(np.abs(-scan_positions[:, 1] * pixel_size - (-init_positions_x)))
            y_diff = np.mean(np.abs(scan_positions[:, 0] * pixel_size - init_positions_y))
            if params.get("beam_source", "xray") == "electron":
                plt.xlabel(f"X [{pixel_unit}] (average error: {x_diff * 10:.2f} angstrom)")
                plt.ylabel(f"Y [{pixel_unit}] (average error: {y_diff * 10:.2f} angstrom)")
            else:
                plt.xlabel(f"X [{pixel_unit}] (average error: {x_diff * 1e3:.2f} nm)")
                plt.ylabel(f"Y [{pixel_unit}] (average error: {y_diff * 1e3:.2f} nm)")
            plt.legend(
                ["Initial positions", "Refined positions"],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
            )
            plt.grid(True)
            plt.xlim(pos_x_min * range_factor, pos_x_max * range_factor)
            plt.ylim(pos_y_min * range_factor, pos_y_max * range_factor)
            plt.savefig(f"{recon_path}/positions/positions_Niter{iter}.png", dpi=300)
            plt.close()

        # Plot affine transformation parameters from probe position correction
        affine_matrix = task.reconstructor.parameter_group.probe_positions.affine_transform_matrix
        affine_transform_components = (
            task.reconstructor.parameter_group.probe_positions.affine_transform_components
        )

        # Extract and store transformation parameters
        scale = affine_transform_components["scale"]
        asymmetry = affine_transform_components["asymmetry"]
        rotation = affine_transform_components["rotation"] * 180 / np.pi  # Convert to degrees
        shear = affine_transform_components["shear"]

        pos_scale.append(scale)
        pos_assymetry.append(asymmetry)
        pos_rotation.append(rotation)
        pos_shear.append(shear)
        iterations.append(iter)

        # Define parameter info for streamlined plotting
        param_info = [
            ("Scale", pos_scale, "Scale Factor", (0.97, 1.03), None),
            ("Asymmetry", np.array(pos_assymetry) * 100.0, "Asymmetry (%)", None, None),
            ("Rotation", pos_rotation, "Rotation (deg)", None, None),
            ("Shear", pos_shear, "Shear", None, None),
        ]

        if params.get("save_plots", True):
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            axs = axs.ravel()
            for i, (title, data, ylabel, ylim, color) in enumerate(param_info):
                axs[i].plot(iterations, data, "o-", color="blue")
                axs[i].set_xlabel("Iterations")
                axs[i].set_ylabel(ylabel)
                axs[i].set_title(title)
                axs[i].grid(True)
                if ylim:
                    axs[i].set_ylim(*ylim)

            plt.tight_layout()
            plt.savefig(f"{recon_path}/positions_affine/positions_affine_Niter{iter}.png", dpi=300)
            plt.close(fig)

    # Plot loss vs iterations
    loss = task.reconstructor.loss_tracker.table["loss"]
    
    if params.get("save_plots", True):
        plt.figure()
        plt.plot(loss, label="Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        loss_array = loss.values
        min_loss = min(loss_array)
        final_loss = loss_array[-1] if len(loss_array) > 0 else 0
        plt.title(f"Min Loss: {min_loss:.2e}. Final Loss: {final_loss:.2e}")
        plt.savefig(f"{recon_path}/loss/loss_Niter{iter}.png", dpi=300)
        plt.close()

    # Save results in hdf5 format
    init_positions_px = np.array([init_positions_y / pixel_size, init_positions_x / pixel_size]).T

    with h5py.File(f"{recon_path}/recon_Niter{iter}.h5", "w") as hdf_file:
        hdf_file.create_dataset("probe", data=recon_probe)
        hdf_file.create_dataset("object", data=recon_object)
        hdf_file.create_dataset("loss", data=loss)
        hdf_file.create_dataset("positions_px", data=scan_positions)
        hdf_file.create_dataset("init_positions_px", data=init_positions_px)
        hdf_file.create_dataset("obj_pixel_size_m", data=task.object_options.pixel_size_m)
        if params["position_correction"]:
            pos_corr_group = hdf_file.create_group("pos_corr")
            pos_corr_group.create_dataset("scale", data=pos_scale)
            pos_corr_group.create_dataset("asymmetry", data=pos_assymetry)
            pos_corr_group.create_dataset("rotation", data=pos_rotation)
            pos_corr_group.create_dataset("shear", data=pos_shear)
            pos_corr_group.create_dataset("iterations", data=iterations)
            pos_corr_group.create_dataset("affine_matrix", data=affine_matrix.cpu().numpy())
        if recon_probe.shape[0] > 1:
            hdf_file.create_dataset("opr_mode_weights", data=opr_mode_weights)
        if recon_object_roi.shape[0] > 1:  # multislice recon
            slice_spacings = task.object_options.slice_spacings_m
            hdf_file.create_dataset("slice_spacings_m", data=slice_spacings)

    if params["number_of_iterations"] == iter:
        if params.get("collect_object_phase", False):  # copy final object phase to a collection folder
            obj_ph_collection_dir = os.path.join(
                params["data_directory"],
                "ptychi_recons",
                params["recon_parent_dir"],
                "object_ph_collection",
            )
            os.makedirs(obj_ph_collection_dir, exist_ok=True)
            verbose_print(
                f"\nSaving final object phase to {obj_ph_collection_dir}/S{params['scan_num']:04d}.tiff",
                print_mode
            )
            shutil.copyfile(
                f"{recon_path}/object_ph/object_ph_Niter{iter}.tiff",
                f"{obj_ph_collection_dir}/S{params['scan_num']:04d}.tiff"
            )

        if params.get("collect_probe_magnitude", False):  # copy final probe magnitude to a collection folder
            probe_mag_collection_dir = os.path.join(
                params["data_directory"],
                "ptychi_recons",
                params["recon_parent_dir"],
                "probe_mag_collection",
            )
            os.makedirs(probe_mag_collection_dir, exist_ok=True)
            verbose_print(
                f"\nSaving final probe magnitude to {probe_mag_collection_dir}/S{params['scan_num']:04d}.tiff",
                print_mode
            )
            shutil.copyfile(
                f"{recon_path}/probe_mag/probe_mag_Niter{iter}.tiff",
                f"{probe_mag_collection_dir}/S{params['scan_num']:04d}.tiff"
            )

def create_reconstruction_path(params, options):
    # Check if user has specified a custom reconstruction path to overwrite the default
    if params.get("recon_dir_base", ''):
        recon_dir_base = params["recon_dir_base"]
    else:
        recon_dir_base = os.path.join(
            params["data_directory"],
            "ptychi_recons",
            params["recon_parent_dir"],
            f"S{params['scan_num']:04d}",
        )

    # Append batching mode to the path
    batching_mode_suffix = {
        api.BatchingModes.RANDOM: "r",
        api.BatchingModes.UNIFORM: "s",
        api.BatchingModes.COMPACT: "c",
    }.get(options.reconstructor_options.batching_mode, "")

    recon_path = (
        recon_dir_base
        + f"/Ndp{options.data_options.data.shape[1]}_LSQML_{batching_mode_suffix}{options.reconstructor_options.batch_size}"
    )

    if batching_mode_suffix == "c" and options.reconstructor_options.momentum_acceleration_gain > 0:
        recon_path += f"_m{options.reconstructor_options.momentum_acceleration_gain}"

    if params.get("noise_model", "gaussian") == "poisson":
        recon_path += f"_poisson"
    else:
        recon_path += f"_gaussian"

    if params.get("near_field_ptycho", False):
        recon_path += f"_nf_fsd{params['focal_sample_dist_m'] / 1e-3:.2f}mm"
        
    recon_path += f"_p{options.probe_options.initial_guess.shape[1]}"

    # Append optional parameters to the path
    if options.probe_options.center_constraint.enabled:
        recon_path += "_cp"
    if options.object_options.multimodal_update:
        recon_path += "_mm"

    if options.opr_mode_weight_options.optimizable:
        recon_path += f"_opr{options.probe_options.initial_guess.shape[0] - 1}"
    if options.opr_mode_weight_options.optimize_intensity_variation:
        recon_path += "_ic"

    if params["object_thickness_m"] > 0 and params["number_of_slices"] > 1:
        if params.get("beam_source", "xray") == "electron":
            recon_path += (
                f"_Ns{params['number_of_slices']}_T{params['object_thickness_m'] / 1e-9:.2f}nm"
            )
        else:
            recon_path += (
                f"_Ns{params['number_of_slices']}_T{params['object_thickness_m'] / 1e-6:.2f}um"
            )
        if (
            options.object_options.multislice_regularization.enabled
            and options.object_options.multislice_regularization.weight > 0
        ):
            recon_path += f"_reg{options.object_options.multislice_regularization.weight}"

    if options.probe_position_options.optimizable:
        recon_path += f"_pc{options.probe_position_options.optimization_plan.start}" 
        if params.get("position_correction_gradient_method", "gaussian") == "gaussian":
            recon_path += "_g"
        elif params.get("position_correction_gradient_method", "gaussian") == "fourier":
            recon_path += "_f"
        if options.probe_position_options.correction_options.update_magnitude_limit > 0:
            recon_path += (
                f"_ul{options.probe_position_options.correction_options.update_magnitude_limit}"
            )
        if options.probe_position_options.correction_options.slice_for_correction:
            recon_path += (
                f"_layer{options.probe_position_options.correction_options.slice_for_correction}"
            )
        if options.probe_position_options.affine_transform_constraint.apply_constraint:
            recon_path += "_affine"

    if params.get("init_probe_propagation_distance_mm", 0) != 0:
        recon_path += f"_pd{params['init_probe_propagation_distance_mm']}"

    if options.object_options.smoothness_constraint.enabled:
        recon_path += f"_smooth{options.object_options.smoothness_constraint.alpha}"

    if options.reconstructor_options.forward_model_options.diffraction_pattern_blur_sigma:
        recon_path += f"_dpBlur{options.reconstructor_options.forward_model_options.diffraction_pattern_blur_sigma}"

    # Check if any diffraction pattern transformations are enabled
    dp_transforms = {
        "up_down": params.get("flip_diffraction_patterns_up_down", False),
        "left_right": params.get("flip_diffraction_patterns_left_right", False),
        "transpose": params.get("transpose_diffraction_patterns", False),
    }

    if any(dp_transforms.values()):
        recon_path += "_dpFlip"
        # Add specific transform indicators to path
        if dp_transforms["up_down"]:
            recon_path += "_ud"
        if dp_transforms["left_right"]:
            recon_path += "_lr"
        if dp_transforms["transpose"]:
            recon_path += "_tr"

    # Append any additional suffix
    if params["recon_dir_suffix"]:
        recon_path += f"_{params['recon_dir_suffix']}"

    # Ensure the directory structure exists (only on rank 0 to avoid race conditions)
    try:
        rank = dist.get_rank()
    except (RuntimeError, ValueError):
        rank = 0
    
    if rank == 0:
        if options.object_options.slice_spacings_m:  # multislice recon
            os.makedirs(os.path.join(recon_path, "object_ph_layers"), exist_ok=True)
            os.makedirs(os.path.join(recon_path, "object_ph_total"), exist_ok=True)
            os.makedirs(os.path.join(recon_path, "object_mag_layers"), exist_ok=True)
            os.makedirs(os.path.join(recon_path, "object_mag_total"), exist_ok=True)
            os.makedirs(os.path.join(recon_path, "probe_propagation_mag"), exist_ok=True)
        else:
            os.makedirs(os.path.join(recon_path, "object_ph"), exist_ok=True)
            os.makedirs(os.path.join(recon_path, "object_mag"), exist_ok=True)

        os.makedirs(os.path.join(recon_path, "probe_mag"), exist_ok=True)
        if options.opr_mode_weight_options.optimizable and params.get(
            "save_probe_at_each_scan_position", False
        ):
            os.makedirs(os.path.join(recon_path, "probe_mag_opr"), exist_ok=True)
        os.makedirs(os.path.join(recon_path, "loss"), exist_ok=True)
        if params["position_correction"]:
            os.makedirs(os.path.join(recon_path, "positions"), exist_ok=True)
            os.makedirs(os.path.join(recon_path, "positions_affine"), exist_ok=True)
        print(f"Reconstruction results will be saved in: {recon_path}")
    
    # Synchronize all ranks to ensure directories are created before proceeding
    if dist.is_initialized():
        dist.barrier()

    return recon_path


def save_initial_conditions(recon_path, params, options):
    # Only save from rank 0 to avoid file locking issues in distributed training
    try:
        rank = dist.get_rank()
    except (RuntimeError, ValueError):
        rank = 0
    
    if rank != 0:
        return
    
    det_pixel_size_m = params["det_pixel_size_m"]
    if params.get("beam_source", "xray") == "electron":
        pixel_size = options.object_options.pixel_size_m * 1e9
        pixel_unit = "nm"
    else:
        pixel_size = options.object_options.pixel_size_m * 1e6
        pixel_unit = "um"

    # save sum of all diffraction patterns
    dp_sum = np.sum(options.data_options.data, axis=0)

    # plt.imsave(f'{recon_path}/dp_sum.png', dp_sum, cmap='jet', metadata={'unit': 'pixel', 'pixel_size': 1})
    # imwrite(f'{recon_path}/dp_sum.tiff', normalize_by_bit_depth(np.sum(options.data_options.data, axis=0), '16'))

    # Apply the jet colormap to convert data to RGB
    norm = plt.Normalize(vmin=dp_sum.min(), vmax=dp_sum.max())
    cmap = plt.cm.jet
    colored_dp_sum = cmap(norm(dp_sum))  # This creates an RGBA array
    colored_dp_sum = (colored_dp_sum[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB uint8

    # Save with ImageJ-compatible resolution information
    imwrite(
        f"{recon_path}/dp_sum.tiff",
        colored_dp_sum,
        photometric="rgb",
        resolution=(1 / pixel_size, 1 / pixel_size),
        metadata={"unit": pixel_unit, "pixel_size": pixel_size},
        imagej=True,
    )

    # Save options to a file
    with open(f"{recon_path}/ptychi_options.pkl", "wb") as f:
        import pickle

        options_dict_temp = options.__dict__.copy()
        if not params["save_diffraction_patterns"]:
            # Store the data temporarily
            data_backup = options_dict_temp["data_options"].data
            # Remove data from the copy that will be saved
            options_dict_temp["data_options"].data = None
            # Save the options without the data
            pickle.dump(options_dict_temp, f)
            # Restore the data
            options_dict_temp["data_options"].data = data_backup
        else:
            pickle.dump(options_dict_temp, f)

    # save initial probe
    init_probe_mag = np.abs(options.probe_options.initial_guess[0].cpu().detach().numpy())
    probe_temp = np.hstack(init_probe_mag)
    N_probe = probe_temp.shape[0]
    # plt.imsave(f'{recon_path}/init_probe_mag.png', probe_temp, cmap='plasma')
    # #imwrite(f'{recon_path}/init_probe_mag.tiff', normalize_by_bit_depth(probe_temp, '16'))

    norm = plt.Normalize(vmin=probe_temp.min(), vmax=probe_temp.max())
    cmap = plt.cm.plasma
    colored_probe_temp = cmap(norm(probe_temp))  # This creates an RGBA array
    colored_probe_temp = (colored_probe_temp[:, :, :3] * 255).astype(
        np.uint8
    )  # Convert to RGB uint8

    # Save with ImageJ-compatible resolution information
    imwrite(
        f"{recon_path}/init_probe_mag.tiff",
        colored_probe_temp,
        photometric="rgb",
        resolution=(1 / pixel_size, 1 / pixel_size),
        metadata={"unit": pixel_unit, "pixel_size": pixel_size},
        imagej=True,
    )

    # Create figure and axis
    # fig, ax = plt.subplots()

    # # Display the image
    # im = ax.imshow(probe_temp, cmap='plasma')

    # # Add scale bar
    # bar_length_pixels = 1e-6/options.object_options.pixel_size_m  # Length of scale bar in pixels
    # bar_width_pixels = N_probe/30    # Width of scale bar in pixels
    # bar_position_x = N_probe/20     # from left edge
    # bar_position_y = N_probe*0.9  # from bottom

    # # Create and add the scale bar
    # scale_bar = Rectangle((bar_position_x, bar_position_y),
    #                     bar_length_pixels, bar_width_pixels,
    #                     fc='white', ec='none')
    # ax.add_patch(scale_bar)

    # # Optional: Add text above scale bar
    # plt.text(bar_position_x + bar_length_pixels/2,
    #         bar_position_y - 3,
    #         '1 um',
    #         color='white',
    #         ha='center',
    #         va='bottom',
    #         fontsize=5)

    # # Remove axes
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_frame_on(False)

    # # Save the figure
    # plt.savefig(f'{recon_path}/init_probe_mag.png',
    #             dpi=500,
    #             bbox_inches='tight',
    #             pad_inches=0)
    # plt.close()

    # plot initial positions
    global \
        pos_x_min, \
        pos_x_max, \
        pos_y_min, \
        pos_y_max, \
        init_positions_y, \
        init_positions_x, \
        range_factor
    if params.get("beam_source", "xray") == "electron":
        init_positions_y = (
            options.probe_position_options.position_y_px * options.object_options.pixel_size_m * 1e9
        )
        init_positions_x = (
            options.probe_position_options.position_x_px * options.object_options.pixel_size_m * 1e9
        )
    else:
        init_positions_y = (
            options.probe_position_options.position_y_px * options.object_options.pixel_size_m * 1e6
        )
        init_positions_x = (
            options.probe_position_options.position_x_px * options.object_options.pixel_size_m * 1e6
        )
    if params.get("save_plots", True):
        plt.figure()
        plt.scatter(-init_positions_x, init_positions_y, s=1, edgecolors="blue")
        plt.xlabel(f"X [{pixel_unit}]")
        plt.ylabel(f"Y [{pixel_unit}]")
        plt.legend(["Initial positions"], loc="upper center", bbox_to_anchor=(0.5, 1.15))
        plt.grid(True)
        pos_x_min, pos_x_max = plt.xlim()
        pos_y_min, pos_y_max = plt.ylim()

        range_factor = 1.1
        plt.xlim(pos_x_min * range_factor, pos_x_max * range_factor)
        plt.ylim(pos_y_min * range_factor, pos_y_max * range_factor)
        plt.savefig(f"{recon_path}/init_positions.png", dpi=300)
        plt.close()

    if params["position_correction"]:
        global pos_scale, pos_assymetry, pos_rotation, pos_shear, iterations
        pos_scale = []
        pos_assymetry = []
        pos_rotation = []
        pos_shear = []
        iterations = []
    # Save parameters to a JSON file in the reconstruction path
    params_file_path = f"{recon_path}/pear_params.json"
    with open(params_file_path, "w") as params_file:
        json.dump(params, params_file, indent=4)


def initialize_recon(params):
    instrument = params["instrument"].lower()
    dp_Npix = params["diff_pattern_size_pix"]
    energy = params["beam_energy_kev"]
    global print_mode
    print_mode = params.get('print_mode', 'debug')

    # Load diffraction patterns and positions
    try:
        if params.get("load_processed_hdf5") or instrument == "simu":
            h5_dp_path = find_matching_recon(
                params.get("path_to_processed_hdf5_dp"), params["scan_num"]
            )
            h5_pos_path = find_matching_recon(
                params.get("path_to_processed_hdf5_pos"), params["scan_num"]
            )
            dp, positions_m = _load_data_hdf5(h5_dp_path, h5_pos_path, dp_Npix, params.get("diff_pattern_center_x"), params.get("diff_pattern_center_y"))
        else:
            dp, positions_m = _load_data_raw(
                instrument,
                params.get("data_directory"),
                params.get("scan_num"),
                dp_Npix,
                params.get("diff_pattern_center_x"),
                params.get("diff_pattern_center_y"),
            )
    except Exception as e:
        print(f"Error loading diffraction patterns and positions")
        raise e

    verbose_print(f"Shape of diffraction patterns: {dp.shape}", print_mode)

    if dp.shape[0] < params.get("minimal_num_of_diff_pattern", 1):
        raise ValueError(f"Too few diffraction patterns: {dp.shape[0]} < {params.get('minimal_num_of_diff_pattern', 1)}")

    if dp.shape[1] < dp_Npix:
        dp = crop_pad(dp, (dp_Npix, dp_Npix))
        verbose_print(f"Pad diffraction patterns with zeros to: {dp.shape}", print_mode)

    # Process diffraction patterns with orientation transforms if specified
    dp = _process_diffraction_patterns(dp, params)

    # Load external positions
    if params["path_to_init_positions"]:
        positions_m = _prepare_initial_positions(params)

    positions_m = _apply_affine_transform(positions_m, params)

    # Center positions so that max positive value equals max negative value
    if params.get("center_init_positions", True):
        positions_m[:, 0] = positions_m[:, 0] - (np.max(positions_m[:, 0]) + np.min(positions_m[:, 0])) / 2
        positions_m[:, 1] = positions_m[:, 1] - (np.max(positions_m[:, 1]) + np.min(positions_m[:, 1])) / 2

    params["det_pixel_size_m"] = (
        75e-6
        if instrument
        in ["velo", "velociprobe", "bnp", "bionanoprobe", "2ide", "2xfm", "lynx", "simu"]
        else 172e-6
    )

    if params.get("beam_source", "xray") == "electron":
        params["wavelength_m"] = 12.3986 / np.sqrt((2 * 511.0 + energy) * energy) / 1e10
        verbose_print(f"Wavelength (angstrom): {params['wavelength_m'] * 1e10:.3f}", print_mode)

        # p.dx_spec = 1./p.asize./(p.d_alpha/1e3/p.lambda); %angstrom
        params["obj_pixel_size_m"] = 1 / dp_Npix / params.get("dk", 1) / 1e10  # pixel size
        verbose_print(
            f"Pixel size of reconstructed object (angstrom): {params['obj_pixel_size_m'] * 1e10:.3f}",
            print_mode
        )
        obj_pad_size = params.get("obj_pad_size_m", 1e-9)

    else:
        params["wavelength_m"] = 1.23984193e-9 / energy
        verbose_print(f"Wavelength (nm): {params['wavelength_m'] * 1e9:.3f}", print_mode)
        if params.get("near_field_ptycho", False):
            params["nearfield_magnification"] = (params["det_sample_dist_m"] - params["focal_sample_dist_m"]) / params["focal_sample_dist_m"]
            params["obj_pixel_size_m"] = params["det_pixel_size_m"] / params["nearfield_magnification"]
            params["det_sample_dist_m"] = params["det_sample_dist_m"] / params["nearfield_magnification"]
        else:
            params["obj_pixel_size_m"] = (
                params["wavelength_m"]
                * params["det_sample_dist_m"]
                / params["det_pixel_size_m"]
                / dp_Npix
            )  # pixel size
        verbose_print(f"Pixel size of reconstructed object (nm): {params['obj_pixel_size_m'] * 1e9:.3f}", print_mode)
        obj_pad_size = params.get("obj_pad_size_m", 1e-6)

    # Find clusters of scan positions where distances between points are smaller than 20 nm
    if params.get("burst_ptycho", False):
        verbose_print(
            f"Performing burst ptycho clustering assuming threshold of {params['obj_pixel_size_m'] * 1e9:.3f} nm.",
            print_mode
        )
        from sklearn.cluster import DBSCAN

        # Apply DBSCAN clustering
        # eps is the maximum distance between two samples to be considered in the same cluster
        # min_samples is the minimum number of samples in a neighborhood to form a core point
        clustering = DBSCAN(eps=params["obj_pixel_size_m"], min_samples=1).fit(positions_m)

        # Get cluster labels for each position
        cluster_labels = clustering.labels_

        # Count number of clusters (excluding noise points with label -1)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

        # Count points in each cluster
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)

        verbose_print(f"Found {n_clusters} position clusters.", print_mode)
        for i, label in enumerate(unique_labels):
            if label == -1:
                verbose_print(f"  Noise points: {counts[i]}", print_mode)
            else:
                # print(f"  Cluster {label}: {counts[i]} positions")
                pass

        # Create a new positions array with one position per cluster (using cluster centroids)
        # This will reduce the number of scan points to the number of clusters

        # Initialize the new positions array
        positions_m_clustered = np.zeros((n_clusters, 2))

        # For each cluster, calculate the centroid (average position)
        for i, label in enumerate(unique_labels):
            if label != -1:  # Skip noise points (if any)
                # Get all positions in this cluster
                cluster_positions = positions_m[cluster_labels == label]

                # Calculate the average position (centroid)
                cluster_centroid = np.mean(cluster_positions, axis=0)

                # Store the centroid in the new positions array
                positions_m_clustered[label] = cluster_centroid

        verbose_print(f"Created new positions array with {n_clusters} points (one per cluster)", print_mode)
        verbose_print(
            f"Original positions shape: {positions_m.shape}, Clustered positions shape: {positions_m_clustered.shape}",
            print_mode
        )

        # Optionally, replace the original positions with the clustered ones
        # Uncomment the next line to use the clustered positions instead of the original ones
        positions_m = positions_m_clustered

        # Average diffraction patterns for each cluster
        verbose_print("Averaging diffraction patterns within each cluster...", print_mode)

        # Create a new array to store the averaged diffraction patterns
        dp_shape = dp.shape[1:]  # Get the shape of a single diffraction pattern
        dp_clustered = np.zeros((n_clusters, *dp_shape), dtype=dp.dtype)

        # For each cluster, average the diffraction patterns
        for i, label in enumerate(unique_labels):
            if label != -1:  # Skip noise points (if any)
                # Get indices of all positions in this cluster
                cluster_indices = np.where(cluster_labels == label)[0]

                # Extract diffraction patterns for this cluster
                cluster_dps = dp[cluster_indices]

                # Average the diffraction patterns
                cluster_dp_avg = np.mean(cluster_dps, axis=0)

                # Store the averaged diffraction pattern
                dp_clustered[label] = cluster_dp_avg

                # print(f"  Cluster {label}: Averaged {len(cluster_indices)} diffraction patterns")

        verbose_print(f"Original dp shape: {dp.shape}, Clustered dp shape: {dp_clustered.shape}", print_mode)

        # Replace the original diffraction patterns with the clustered ones
        dp = dp_clustered

        # Store cluster information in params for later use
        params["burst_mode_clusters"] = {
            "labels": cluster_labels.tolist(),
            "count": n_clusters,
            "threshold_m": params.get("burst_clustering_threshold_m"),
        }

    init_positions_px = positions_m / params["obj_pixel_size_m"]

    # Check if positions contain NaN values
    if np.isnan(init_positions_px).any():
        verbose_print(
            f"WARNING: Initial positions contain {np.sum(np.isnan(init_positions_px))} NaN values!",
            print_mode
        )
        verbose_print(f"Initial positions shape: {init_positions_px.shape}", print_mode)
    else:
        verbose_print(f"Initial positions shape: {init_positions_px.shape}, no NaN values detected", print_mode)

    # Load initial probe
    init_probe = _prepare_initial_probe(dp, params)

    # Load initial object
    init_object = _prepare_initial_object(
        params,
        init_positions_px,
        init_probe.shape[-2:],
        round(obj_pad_size / params["obj_pixel_size_m"]),
    )

    return (dp, init_positions_px, init_probe, init_object, params)


def _load_data_raw(instrument, base_path, scan_num, dp_Npix, dp_cen_x, dp_cen_y):
    instrument_loaders = {
        "velo": load_data_velo,
        "velociprobe": load_data_velo,
        "bnp": load_data_bnp,
        "bionanoprobe": load_data_bnp,
        "12idc": load_data_12idc,
        "2xfm": load_data_2xfm,
        "2ide": load_data_2xfm,
        "lynx": load_data_lynx,
    }
    instrument = instrument.lower()
    if instrument not in instrument_loaders:
        raise ValueError(f"Unsupported instrument: {instrument}")

    dp, positions = instrument_loaders[instrument](base_path, scan_num, dp_Npix, dp_cen_x, dp_cen_y, print_mode)

    return dp, positions

def _prepare_initial_positions(params):
    params["path_to_init_positions"] = find_matching_recon(
        params["path_to_init_positions"], params["scan_num"]
    )
    verbose_print("Loading initial positions from a ptychi reconstruction at:", print_mode)
    verbose_print(params["path_to_init_positions"], print_mode)
    positions_px = _load_ptychi_recon(params["path_to_init_positions"], "positions_px")
    input_obj_pixel_size = _load_ptychi_recon(params["path_to_init_positions"], "obj_pixel_size_m")

    return positions_px * input_obj_pixel_size


def _apply_affine_transform(positions_m, params):
    affine_matrix = np.array(params.get("init_position_affine_matrix", np.eye(2)))
    verbose_print(f"Affine matrix for initial positions: {affine_matrix.flatten()}", print_mode)
    transformed_positions_m = positions_m @ affine_matrix

    return transformed_positions_m


def _prepare_initial_object(params, positions_px, probe_size, extra_size):
    if params["path_to_init_object"]:
        params["path_to_init_object"] = find_matching_recon(
            params["path_to_init_object"], params["scan_num"]
        )

        verbose_print("Loading initial object from a ptychi reconstruction at:", print_mode)
        verbose_print(params["path_to_init_object"], print_mode)
        init_object = _load_ptychi_recon(params["path_to_init_object"], "object")
        verbose_print(f"Initial object shape: {init_object.shape}", print_mode)
        input_obj_pixel_size = _load_ptychi_recon(params["path_to_init_object"], "obj_pixel_size_m")

        # Handle multislice object initialization
        if init_object.shape[0] > 1:  # input object is a multislice reconstruction
            # Step 1: Select specific layers if specified
            layer_select = params.get("init_layer_select", [])
            if layer_select:
                # Filter out invalid layer indices
                layer_select = [i for i in layer_select if 0 <= i < init_object.shape[0]]
                if layer_select:
                    verbose_print(f"Selecting specific layers: {layer_select}", print_mode)
                    init_object = init_object[layer_select]
                else:
                    verbose_print("No valid layers specified in init_layer_select, using all layers", print_mode)

            # Step 2: Pre-process layers based on specified mode
            init_layer_preprocess = params.get("init_layer_preprocess", "")
            if init_layer_preprocess == "avg":
                # Average all layers but keep the same number of layers
                verbose_print("Averaging initial layers", print_mode)
                obj_avg = np.prod(init_object, axis=0)
                # Unwrap phase and divide by number of layers
                obj_avg_phase = (
                    unwrap_phase_2d(
                        torch.from_numpy(obj_avg).cuda(),
                        image_grad_method="fourier_differentiation",
                        image_integration_method="fourier",
                    )
                    .cpu()
                    .numpy()
                )
                obj_avg_phase = obj_avg_phase / init_object.shape[0]
                obj_avg = np.abs(obj_avg) * np.exp(1j * obj_avg_phase)
                # Replicate the averaged layer
                init_object = np.repeat(obj_avg[np.newaxis, :, :], init_object.shape[0], axis=0)

            elif init_layer_preprocess == "avg1":
                # Average all layers and keep only one layer
                verbose_print("Averaging initial layers and keeping only one", print_mode)
                obj_avg = np.prod(init_object, axis=0)
                # Unwrap phase and divide by number of layers
                obj_avg_phase = (
                    unwrap_phase_2d(
                        torch.from_numpy(obj_avg).cuda(),
                        image_grad_method="fourier_differentiation",
                        image_integration_method="fourier",
                    )
                    .cpu()
                    .numpy()
                )
                obj_avg_phase = obj_avg_phase / init_object.shape[0]
                obj_avg = np.abs(obj_avg) * np.exp(1j * obj_avg_phase)
                init_object = obj_avg[np.newaxis, :, :]

            elif init_layer_preprocess == "interp" and "init_layer_interp" in params:
                # Interpolate layers to new z positions
                interp_positions = params["init_layer_interp"]
                verbose_print(
                    f"Interpolating {init_object.shape[0]} initial layers to {len(interp_positions)} layers",
                    print_mode
                )

                # Create interpolation function for real and imaginary parts separately
                real_interp = interp1d(
                    np.arange(init_object.shape[0]),
                    init_object.real,
                    axis=0,
                    kind="cubic",
                    bounds_error=False,
                    fill_value="extrapolate",
                )

                imag_interp = interp1d(
                    np.arange(init_object.shape[0]),
                    init_object.imag,
                    axis=0,
                    kind="cubic",
                    bounds_error=False,
                    fill_value="extrapolate",
                )

                # Interpolate to new positions
                real_part = real_interp(interp_positions)
                imag_part = imag_interp(interp_positions)

                # Combine real and imaginary parts
                init_object = real_part + 1j * imag_part

            # Step 3: Add or remove layers based on target number of slices
            target_layers = params["number_of_slices"]

            if init_object.shape[0] > target_layers:
                verbose_print(
                    f"Initial object has more layers ({init_object.shape[0]}) than target ({target_layers})",
                    print_mode
                )
                if target_layers == 1:
                    # If only one layer is needed, use the product of all layers
                    verbose_print("Using product of all layers for single-slice reconstruction", print_mode)
                    obj_prod = np.prod(init_object, axis=0)
                    init_object = obj_prod[np.newaxis, :, :]
                else:
                    # Select middle layers
                    verbose_print(f"Selecting middle {target_layers} layers", print_mode)
                    start_idx = (init_object.shape[0] - target_layers) // 2
                    end_idx = start_idx + target_layers
                    init_object = init_object[start_idx:end_idx]

            elif init_object.shape[0] < target_layers:
                # Need to add more layers
                n_add = target_layers - init_object.shape[0]
                verbose_print(f"Adding {n_add} more layers to initial {init_object.shape[0]} layers", print_mode)

                append_mode = params.get("init_layer_append_mode", "edge")

                if append_mode == "avg":
                    # Use averaged layer for padding
                    obj_avg = np.prod(init_object, axis=0)
                    obj_avg_phase = (
                        unwrap_phase_2d(
                            torch.from_numpy(obj_avg).cuda(),
                            image_grad_method="fourier_differentiation",
                            image_integration_method="fourier",
                        )
                        .cpu()
                        .numpy()
                    )
                    obj_avg_phase = obj_avg_phase / init_object.shape[0]
                    obj_avg = np.abs(obj_avg) * np.exp(1j * obj_avg_phase)
                    obj_pre = obj_post = obj_avg

                elif append_mode == "edge":
                    # Use first/last layer for padding
                    obj_pre = init_object[0]
                    obj_post = init_object[-1]

                else:  # 'vac' or default
                    # Use vacuum (ones) for padding
                    verbose_print(f"Pad input object with vacuum (ones) layers", print_mode)
                    obj_shape = init_object.shape[1:]
                    obj_pre = obj_post = np.ones(obj_shape, dtype=np.complex64)
                # Add layers alternating between front and back
                new_object = init_object.copy()
                for i in range(n_add):
                    if i % 2 == 0:
                        # Add to end
                        new_object = np.concatenate([new_object, obj_post[np.newaxis, :, :]])
                    else:
                        # Add to beginning
                        new_object = np.concatenate([obj_pre[np.newaxis, :, :], new_object])

                init_object = new_object

            # Step 4: Apply scaling factor to phase if specified
            scaling_factor = params.get("init_layer_scaling_factor", 1.0)
            if scaling_factor != 1.0:
                verbose_print(f"Scaling layer phases by factor {scaling_factor}", print_mode)
                for i in range(init_object.shape[0]):
                    layer = init_object[i]
                    # Unwrap phase and scale
                    layer_phase = (
                        unwrap_phase_2d(
                            torch.from_numpy(layer).cuda(),
                            image_grad_method="fourier_differentiation",
                            image_integration_method="fourier",
                        )
                        .cpu()
                        .numpy()
                    )
                    layer_phase *= scaling_factor
                    # Recombine amplitude and scaled phase
                    init_object[i] = np.abs(layer) * np.exp(1j * layer_phase)

        # Resize object if pixel size doesn't match
        if input_obj_pixel_size != params["obj_pixel_size_m"]:
            verbose_print(
                f"Input object's pixel size ({input_obj_pixel_size * 1e9:.3f} nm) does not match the expected pixel size ({params['obj_pixel_size_m'] * 1e9:.3f} nm).",
                print_mode
            )
            verbose_print(f"Resizing input object to match the current reconstruction.", print_mode)

            # Calculate zoom factor based on pixel size ratio
            zoom_factor = input_obj_pixel_size / params["obj_pixel_size_m"]

            # Get target shape after zooming first slice
            target_shape = (
                int(init_object.shape[-2] * zoom_factor),
                int(init_object.shape[-1] * zoom_factor),
            )

            # Use resize_complex_array to resize the object
            init_object = resize_complex_array(init_object, new_shape=target_shape)

            # Convert to tensor
        init_object = to_tensor(init_object)

    else:
        verbose_print("Generating a random initial object.", print_mode)
        init_object = torch.ones(
            [
                params["number_of_slices"],
                *get_suggested_object_size(positions_px, probe_size, extra=extra_size),
            ],
            dtype=get_default_complex_dtype(),
        )
        init_object = init_object + 1j * torch.rand(*init_object.shape) * 1e-3

    verbose_print(f"Shape of initial object: {init_object.shape}", print_mode)
    return init_object


def _prepare_initial_probe(dp, params):
    num_probe_modes = params.get("number_probe_modes")
    num_opr_modes = params.get("number_opr_modes")

    if params.get("use_model_FZP_probe", False):
        verbose_print("Generating a model FZP probe.", print_mode)
        if params["instrument"].lower() == "velo" or params["instrument"].lower() == "velociprobe":
            dRn = 50e-9
            Rn = 90e-6
            D_H = 60e-6
            D_FZP = 250e-6
        elif params["instrument"].lower() == "ptycho_probe":
            dRn = 15e-9
            Rn = 90e-6
            D_H = 15e-6
            D_FZP = 250e-6
        else:
            dRn = 50e-9
            Rn = 90e-6
            D_H = 60e-6
            D_FZP = 250e-6
        N_probe_orig = dp.shape[-2] * params["obj_pixel_size_m"] / 4e-9
        # Round N_probe_orig up to the nearest power of 2 for faster FFT
        N_probe_orig = int(2 ** np.ceil(np.log2(N_probe_orig)))

        probe_orig = make_fzp_probe(
            N_probe_orig, params["wavelength_m"], 4e-9, 0, Rn, dRn, D_FZP, D_H
        )
        probe = resize_complex_array(
            probe_orig,
            zoom_factor=(4e-9 / params["obj_pixel_size_m"], 4e-9 / params["obj_pixel_size_m"]),
        )
        # Crop probe to match the diffraction pattern size
        if probe.shape[-1] > dp.shape[1]:
            # Calculate the center of the probe
            center_y, center_x = probe.shape[0] // 2, probe.shape[1] // 2
            # Calculate the half-width of the target size
            half_height, half_width = dp.shape[1] // 2, dp.shape[1] // 2
            # Crop the probe around its center
            probe = probe[
                center_y - half_height : center_y + half_height,
                center_x - half_width : center_x + half_width,
            ]
            # print(f"Probe cropped from {probe.shape[0]}x{probe.shape[1]} to {dp.shape[1]}x{dp.shape[1]}")
    else:
        path_to_init_probe = params.get("path_to_init_probe")
        path_to_init_probe = find_matching_recon(path_to_init_probe, params["scan_num"])
        if path_to_init_probe.endswith(".mat"):
            verbose_print("Loading initial probe from a foldslice reconstruction at:", print_mode)
            verbose_print(path_to_init_probe, print_mode)
            probe = _load_probe_foldslice(path_to_init_probe)
        elif params.get("path_to_init_probe").endswith(".h5"):
            verbose_print("Loading initial probe from a ptychi reconstruction at:", print_mode)
            verbose_print(path_to_init_probe, print_mode)
            probe = _load_ptychi_recon(path_to_init_probe, "probe")
        else:
            raise ValueError(
                "Unsupported file format for initial probe. Only .mat and .h5 files are supported."
            )

    verbose_print(f"Shape of input probe: {probe.shape}", print_mode)

    # TODO: load opr weights too
    if probe.ndim == 4:
        probe = probe[0]
    if probe.ndim == 2:
        probe = probe[None, :, :]  # add incoherent mode dimension

    #   p = options.probe_options.initial_guess[0:1]
    # probe = orthogonalize_initial_probe(to_tensor(probe))
    # p = add_additional_opr_probe_modes_to_probe(to_tensor(p), 2)

    # Assuming probe is a [n_incoherent_modes, h, w]
    if probe.shape[0] >= num_probe_modes:
        probe = probe[:num_probe_modes, :, :]
    else:
        probe_temp = np.zeros(
            (num_probe_modes, probe.shape[-2], probe.shape[-1]), dtype=np.complex64
        )
        probe_temp[: probe.shape[0], :, :] = probe
        probe_temp[probe.shape[0] :, :, :] = probe[-1, :, :]
        # probe_temp[probe.shape[0]:,:,:] = probe[0,:,:]
        # probe_temp[-1,:,:] = probe[0,:,:]
        probe = probe_temp
    
    probe_shifts = params.get("init_probe_shifts_pix", [0, 0])
    # Apply initial probe shifts if specified
    if any(s != 0 for s in probe_shifts):
        verbose_print(f"Applying initial probe shifts: {probe_shifts}", print_mode)
        # probe_shifts: [shift_y, shift_x]
        for i in range(probe.shape[0]):
            probe[i] = scipy.ndimage.shift(
                probe[i], shift=(probe_shifts[0], probe_shifts[1]), mode="nearest", order=1, prefilter=True
            )

    # probe = probe.transpose(0, 2, 1)
    # TODO: determine zoom factor based on pixel size ratio
    if probe.shape[-1:] != dp.shape[-1:]:
        verbose_print(
            f"Resizing probe ({probe.shape[-1]}) to match the diffraction pattern size ({dp.shape[-1]}).",
            print_mode
        )
        probe = resize_complex_array(probe, new_shape=(dp.shape[-2], dp.shape[-1]))

    # Propagate probe if a propagation distance is specified
    propagation_distance_mm = params.get("init_probe_propagation_distance_mm", 0)
    if propagation_distance_mm != 0:
        from ptychi.pear_utils import near_field_evolution

        extent = probe.shape[-1] * params["obj_pixel_size_m"]
        # Convert mm to meters for propagation
        propagation_distance_m = propagation_distance_mm * 1e-3

        # Log the propagation operation
        verbose_print(f"Propagating initialprobe by {propagation_distance_mm} mm", print_mode)

        # Propagate each probe mode
        for i in range(probe.shape[0]):
            probe[i], _, _, _ = near_field_evolution(
                probe[i], propagation_distance_m, params["wavelength_m"], extent, use_ASM_only=True
            )

    # Add OPR mode dimension
    probe = probe[None, ...]
    if params.get("orthogonalize_initial_probe", True):
        verbose_print("Orthogonalizing initial probe", print_mode)
        probe = orthogonalize_initial_probe(to_tensor(probe))

    # Add n_opr_modes - 1 eigenmodes which are randomly initialized
    probe = add_additional_opr_probe_modes_to_probe(to_tensor(probe), num_opr_modes)

    verbose_print(f"Shape of probe after preprocessing: {probe.shape}", print_mode)

    return probe


def _load_probe_foldslice(recon_file):
    # print(f"Attempting to load probe from: {recon_file}")

    try:
        with h5py.File(recon_file, "r") as hdf_file:
            probes = hdf_file["probe"][:]
            probes = np.swapaxes(probes, -1, -2)

            # probes = probes.transpose(*{4: (0,1,3,2), 3: (0,2,1), 2: (1,0)}.get(probes.ndim, range(probes.ndim)))
    except:
        import scipy.io

        verbose_print(f"Attempting to load probe using scipy.io.loadmat", print_mode)
        mat_contents = scipy.io.loadmat(recon_file)
        if "probe" in mat_contents:
            probes = mat_contents["probe"]

        verbose_print(f"Shape of input probe: {probes.shape}", print_mode)
        if probes.ndim == 4:
            probes = probes[..., 0]
            verbose_print(f"Taking the primary OPR mode: {probes.shape}", print_mode)
            verbose_print("Transposing probe to (n_probe_modes, h, w)", print_mode)
            probes = probes.transpose(2, 0, 1)
        elif probes.ndim == 3:
            verbose_print("Transposing probe to (n_probe_modes, h, w)", print_mode)
            probes = probes.transpose(2, 0, 1)
        else:
            probes = probes.transpose(1, 0)

    # print("Shape of probes:", probes.shape)
    if probes.dtype == [
        ("real", "<f8"),
        ("imag", "<f8"),
    ]:  # For mat v7.3, the complex128 is read as this complicated datatype via h5py
        # print(f"Loaded object.dtype = {object.dtype}, cast it to 'complex128'")
        probes = probes.view("complex128")
    return probes


def _load_ptychi_recon(recon_file, variable_name):
    with h5py.File(recon_file, "r") as hdf_file:
        # Check if the dataset is a scalar or not
        dataset = hdf_file[variable_name]
        if dataset.shape == ():  # It's a scalar
            array = dataset[()]  # Use [()] for scalar datasets
        else:
            array = dataset[:]  # Use [:] for non-scalar datasets

    return array


def _normalize_from_zero_to_one(arr):
    norm_arr = (arr - arr.min()) / (arr.max() - arr.min())
    return norm_arr


def normalize_by_bit_depth(arr, bit_depth):
    if bit_depth == "8":
        norm_arr_in_bit_depth = np.uint8(255 * _normalize_from_zero_to_one(arr))
    elif bit_depth == "16":
        norm_arr_in_bit_depth = np.uint16(65535 * _normalize_from_zero_to_one(arr))
    elif bit_depth == "32":
        norm_arr_in_bit_depth = np.float32(_normalize_from_zero_to_one(arr))
    elif bit_depth == "raw":
        norm_arr_in_bit_depth = np.float32(arr)
    else:
        print(
            f"Unsuported bit_depth :{bit_depth} was passed into `result_modes`, `raw` is used instead"
        )
        norm_arr_in_bit_depth = np.float32(arr)

    return norm_arr_in_bit_depth

def _load_data_hdf5(h5_dp_path, h5_position_path, dp_Npix, cen_x, cen_y):
    # if h5_dp_path == h5_position_path: # assume it's a ptychodus product
    #     print("Loading processed scan positions and diffraction patterns in ptychodus convention.")
    # positions = np.stack([f_meta['probe_position_y_m'][...], f_meta['probe_position_x_m'][...]], axis=1)
    #     pixel_size_m = f_meta['object'].attrs['pixel_height_m']
    #     positions_px = positions / pixel_size_m
    #     if subtract_position_mean:
    #         positions_px -= positions_px.mean(axis=0)

    # else: # assume foldslice convention
    verbose_print("Loading processed scan positions and diffraction patterns in foldslice convention.", print_mode)
    verbose_print("Diffraction patterns file:", print_mode)
    verbose_print(h5_dp_path, print_mode)
    dp = h5py.File(h5_dp_path, "r")["dp"][...]
    det_xwidth = int(dp_Npix / 2)
    
    dp = dp[:, cen_y - det_xwidth : cen_y + det_xwidth, cen_x - det_xwidth : cen_x + det_xwidth]
    dp[dp < 0] = 0

    verbose_print("Scan positions file:", print_mode)
    verbose_print(h5_position_path, print_mode)
    ppY = h5py.File(h5_position_path, "r")["ppY"][:].flatten()
    ppX = h5py.File(h5_position_path, "r")["ppX"][:].flatten()
    ppX = ppX - (np.max(ppX) + np.min(ppX)) / 2
    ppY = ppY - (np.max(ppY) + np.min(ppY)) / 2
    positions = np.stack((ppY, ppX), axis=1)
    # positions = np.stack((ppX, ppY), axis=1) #ELE

    return dp, positions

def _process_diffraction_patterns(dp, params):
    """
    Process diffraction patterns with various orientation transformations.

    Parameters:
    -----------
    dp : numpy.ndarray
        The diffraction patterns array of shape (n_patterns, height, width)
    params : dict
        Dictionary containing processing parameters:
        - dp_flip_ud: bool, flip patterns up-down
        - dp_flip_lr: bool, flip patterns left-right
        - dp_transpose: bool, transpose patterns

    Returns:
    --------
    numpy.ndarray
        Processed diffraction patterns
    """
    # Apply transformations in sequence if specified
    if params.get("flip_diffraction_patterns_up_down", False):
        verbose_print("Flipping diffraction patterns up-down", print_mode)
        # Create a contiguous copy after flipping to avoid negative strides
        dp = np.ascontiguousarray(np.flip(dp, axis=1))

    if params.get("flip_diffraction_patterns_left_right", False):
        verbose_print("Flipping diffraction patterns left-right", print_mode)
        # Create a contiguous copy after flipping to avoid negative strides
        dp = np.ascontiguousarray(np.flip(dp, axis=2))

    if params.get("transpose_diffraction_patterns", False):
        verbose_print("Transposing diffraction patterns", print_mode)
        # Transpose each pattern individually
        dp = np.transpose(dp, axes=(0, 2, 1))

    return dp
