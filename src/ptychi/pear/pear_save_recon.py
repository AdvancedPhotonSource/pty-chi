import os
import h5py
import numpy as np
from tifffile import imwrite
from ptychi.image_proc import unwrap_phase_2d
import ptychi.api as api
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import json
import shutil

from .pear_utils import verbose_print

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
        from .pear_utils import near_field_evolution

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

