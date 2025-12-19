import os, glob, time
import h5py
import hdf5plugin  # for reading raw hdf5 files
import numpy as np
import scipy.ndimage
from scipy.interpolate import interp1d
from ptychi.image_proc import unwrap_phase_2d
from ptychi.utils import (
    add_additional_opr_probe_modes_to_probe,
    to_tensor,
    orthogonalize_initial_probe,
    get_suggested_object_size,
    get_default_complex_dtype,
)
import torch

from .pear_utils import make_fzp_probe, resize_complex_array, find_matching_recon, crop_pad, verbose_print
from .pear_aps_data_loaders import (
    load_data_2xfm,
    load_data_12idc,
    load_data_bnp,
    load_data_lynx,
    load_data_velo
)

# Global variable for print mode
print_mode = 'debug'

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

        verbose_print(f"Original dp shape: {dp.shape}, Clustered dp shape: {dp_clustered.shape}", print_mode=print_mode)

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
        from .pear_utils import near_field_evolution

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

