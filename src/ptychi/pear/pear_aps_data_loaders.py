"""
Data loading functions for different ptychography instruments at APS.

This module contains instrument-specific functions for loading diffraction patterns
and scan positions from various beamlines including 2-ID-E, 12-ID-C, BioNanoProbe, 
LYNX, and Velociprobe.
"""

import os
import glob
import time
import h5py
import hdf5plugin
import numpy as np
import scipy.ndimage
from .pear_utils import verbose_print


def load_data_2xfm(base_path, scan_num, det_Npixel, cen_x, cen_y, x_exclude=-2, print_mode='debug'):
    verbose_print("Loading scan positions and diffraction patterns measured by the XFM instrument at 2IDE.", print_mode)
    from .pear_utils_aps import readMDA

    dp_dir = f"{base_path}/ptycho/"
    filePath = "/entry/data/data"

    # Load scan positions from original file
    MDAfile_path = f"{base_path}/mda/2xfm_{scan_num:04d}.mda"

    # if not os.path.exists(XRFfile_path):
    #    raise FileNotFoundError(f"The XRF file path does not exist: {XRFfile_path}")

    if not os.path.exists(MDAfile_path):
        raise FileNotFoundError(f"The MDA file path does not exist: {MDAfile_path}")

    mda_data = readMDA(MDAfile_path)

    x_pos = np.array(mda_data[2].p[0].data)[1]
    y_pos = np.array(mda_data[1].d[5].data)
    STXM = np.array(mda_data[2].d[1].data)
    Ny, Nx = STXM.shape[0], STXM.shape[1]

    x_pos = x_pos[-x_exclude:]
    x_pos -= x_pos.mean()
    y_pos -= y_pos.mean()
    x_pos *= 1e-3
    y_pos *= 1e-3

    N_scan_x = x_pos.shape[0]
    N_scan_y = y_pos.shape[0]
    
    verbose_print(f"{x_pos.shape}", print_mode)
    verbose_print(f"N_scan_y={N_scan_y}, N_scan_x={N_scan_x}, N_scan_dp={N_scan_x * N_scan_y}", print_mode)

    # Load diffraction patterns
    index_x_lb, index_x_ub = int(cen_x - det_Npixel // 2), int(cen_x + (det_Npixel + 1) // 2)
    index_y_lb, index_y_ub = int(cen_y - det_Npixel // 2), int(cen_y + (det_Npixel + 1) // 2)

    dp, scan_posx, scan_posy = [], [], []

    for i in range(N_scan_y):
        verbose_print(f"Loading scan line No.{i + 1}...", print_mode)
        # fileName = data_dir+'fly{:03d}_data_{:03d}.h5'.format(scanNo,i+1+N_scan_y_lb)

        fileName = os.path.join(dp_dir, f"fly{scan_num:03d}_data_{i + 1:03d}.h5")
        with h5py.File(fileName, "r") as h5_data:
            # h5_data = h5py.File(fileName,'r')
            dp_temp = h5_data[filePath][...]
            dp_temp[dp_temp < 0] = 0
            dp_temp[dp_temp > 1e6] = 0
            #print(fileName, dp_temp.shape)

            if dp_temp.shape[0] < 5:
                verbose_print(f"A lot of pixels are missed on this line: {dp_temp.shape[0]} pixels, Skip!", print_mode)
                continue

            dp_crop = dp_temp[-x_exclude:, index_y_lb:index_y_ub, index_x_lb:index_x_ub]
            dp.append(dp_crop)
            scan_posx.extend(x_pos[: dp_crop.shape[0]])
            scan_posy.extend([y_pos[i]] * dp_crop.shape[0])

    positions = np.column_stack((scan_posy, scan_posx))
    dp = np.concatenate(dp, axis=0) if dp else np.array([])  # Concatenate if dp is not empty

    return dp, positions


def load_data_12idc(base_path, scan_num, det_Npixel, cen_x, cen_y, print_mode='debug'):
    """
    Load scan positions and diffraction patterns measured by the Ptycho-SAXS instrument at 12IDC.
    Automatically detects and handles both HDF5 and TIFF file formats.

    Parameters:
    -----------
    base_path : str
        Base directory containing the data
    scan_num : int
        Scan number
    det_Npixel : int
        Number of detector pixels to use
    cen_x : int
        X-coordinate of the center of the detector
    cen_y : int
        Y-coordinate of the center of the detector
    print_mode : str
        Print mode ('debug', 'prod', etc.)

    Returns:
    --------
    tuple
        (diffraction patterns, positions)
    """
    verbose_print(
        "Loading scan positions and diffraction patterns measured by the Ptycho-SAXS instrument at 12IDC.",
        print_mode
    )
    det_xwidth = int(det_Npixel / 2)

    # Check if TIFF files exist for this scan
    tif_dir = os.path.join(base_path, "tifs", f"{scan_num:03d}")
    tif_files = glob.glob(os.path.join(tif_dir, f"*_{scan_num:03d}_*.tif"))

    # Check if processed HDF5 files exist
    ptycho1_dir = os.path.join(base_path, "ptycho1", f"{scan_num:03d}")
    h5_files = glob.glob(os.path.join(ptycho1_dir, f"*_{scan_num:03d}_*.h5"))
    master_file = glob.glob(os.path.join(ptycho1_dir, f"*_{scan_num:03d}_master.h5"))

    # Check if original HDF5 files exist
    ptycho_dir = os.path.join(base_path, "ptycho", f"{scan_num:03d}")
    original_h5_files = glob.glob(os.path.join(ptycho_dir, f"*{scan_num:03d}_*.h5"))

    # Determine which data format to use
    if tif_files:
        verbose_print(f"Found TIFF files in {tif_dir}. Processing TIFF data.", print_mode)
        return load_data_12idc_tiff(base_path, scan_num, det_Npixel, cen_x, cen_y, print_mode)
    elif master_file and h5_files:
        verbose_print(f"Using pre-processed HDF5 files from {ptycho1_dir}", print_mode)
        return load_data_12idc_processed_h5(base_path, scan_num, det_Npixel, cen_x, cen_y, print_mode)
    elif original_h5_files:
        verbose_print(f"Using original HDF5 files from {ptycho_dir}", print_mode)
        return load_data_12idc_original_h5(base_path, scan_num, det_Npixel, cen_x, cen_y, print_mode)
    else:
        raise FileNotFoundError(f"No data files found for scan {scan_num} in any supported format.")


def load_data_12idc_processed_h5(base_path, scan_num, det_Npixel, cen_x, cen_y, print_mode='debug'):
    """Load data from pre-processed HDF5 files"""
    ptycho1_dir = os.path.join(base_path, "ptycho1", f"{scan_num:03d}")
    master_files = glob.glob(os.path.join(ptycho1_dir, f"*_{scan_num:03d}_master.h5"))

    if not master_files:
        raise FileNotFoundError(f"No master file found for scan {scan_num}")

    master_file = master_files[0]
    sample_name = os.path.basename(master_file).split(f"_{scan_num:03d}_")[0]

    # Load data from master file
    with h5py.File(master_file, "r") as h5f:
        # Get beam center from master file if available
        if "beam_center_YX" in h5f.attrs:
            cen_y, cen_x = h5f.attrs["beam_center_YX"]
            verbose_print(f"Using beam center from master file: ({cen_y}, {cen_x})", print_mode)

        # Find all line datasets
        line_datasets = []
        for key in h5f.keys():
            if key.startswith("entry/data/data_"):
                line_datasets.append(key)

        if not line_datasets:
            print(f"Warning: No line datasets found in master file {master_file}")
            # Check if there are any line files directly
            line_files = glob.glob(os.path.join(ptycho1_dir, f"{sample_name}_{scan_num:03d}_*.h5"))
            line_files = [f for f in line_files if not f.endswith("_master.h5")]
            if not line_files:
                raise FileNotFoundError(f"No line files found for scan {scan_num}")
            verbose_print(f"Found {len(line_files)} line files directly in directory", print_mode)
            line_nums = [int(os.path.basename(f).split("_")[-1].split(".")[0]) for f in line_files]
        else:
            line_nums = [int(dataset_path.split("_")[-1]) for dataset_path in line_datasets]

        # Initialize lists to store data
        dp_list = []
        positions_list = []

        # Process each line
        for line_num in line_nums:
            line_file = os.path.join(ptycho1_dir, f"{sample_name}_{scan_num:03d}_{line_num:05d}.h5")
            verbose_print(f"Loading line {line_num} from {line_file}", print_mode)

            if not os.path.exists(line_file):
                verbose_print(f"Warning: Line file not found: {line_file}", print_mode)
                continue

            try:
                with h5py.File(line_file, "r") as line_h5f:
                    # Check if the file has the expected datasets
                    if "/dp" not in line_h5f or "/positions" not in line_h5f:
                        verbose_print(f"Warning: File {line_file} does not contain expected datasets", print_mode)
                        verbose_print(f"Available keys: {list(line_h5f.keys())}", print_mode)
                        continue

                    # Load diffraction patterns
                    dp_data = line_h5f["/dp"][:]

                    # Load positions
                    positions_data = line_h5f["/positions"][:]

                    # Check if data is valid
                    if dp_data.size == 0 or positions_data.size == 0:
                        verbose_print(f"Warning: Empty data in file {line_file}", print_mode)
                        continue

                    # Append to lists
                    dp_list.append(dp_data)
                    positions_list.append(positions_data)
            except Exception as e:
                verbose_print(f"Error loading file {line_file}: {str(e)}", print_mode)
                continue

    # Check if we have any data
    if not dp_list:
        raise ValueError(f"No valid diffraction patterns found for scan {scan_num}")

    # Concatenate data and process positions
    dp = np.concatenate(dp_list, axis=0)
    positions = np.concatenate(positions_list, axis=0)

    # Process positions: extract, invert x, center, and reshape
    positions_processed = np.zeros((len(positions), 2))
    positions_processed[:, 0] = positions[:, 1] * 1e-9 - np.mean(
        positions[:, 1] * 1e-9
    )  # y positions
    positions_processed[:, 1] = -positions[:, 2] * 1e-9 - np.mean(
        -positions[:, 2] * 1e-9
    )  # x positions

    verbose_print(f"Loaded {dp.shape[0]} diffraction patterns and {positions_processed.shape[0]} positions", print_mode)

    # Calculate crop indices
    crop_indices = {
        "x_min": int(cen_x - det_Npixel // 2),
        "x_max": int(cen_x + (det_Npixel + 1) // 2),
        "y_min": int(cen_y - det_Npixel // 2),
        "y_max": int(cen_y + (det_Npixel + 1) // 2),
    }

    # Validate crop dimensions
    if (
        dp.shape[1] < crop_indices["y_max"]
        or dp.shape[2] < crop_indices["x_max"]
        or crop_indices["y_min"] < 0
        or crop_indices["x_min"] < 0
    ):
        raise ValueError(
            f"Diffraction patterns too small to crop to {det_Npixel}x{det_Npixel} with center ({cen_y}, {cen_x})"
        )

    # Crop and clean diffraction patterns
    dp_cropped = dp[
        :,
        crop_indices["y_min"] : crop_indices["y_max"],
        crop_indices["x_min"] : crop_indices["x_max"],
    ]
    dp_cropped = np.clip(dp_cropped, 0, 1e6)  # Replace both operations with a single clip
    return dp_cropped, positions_processed


def load_data_12idc_original_h5(base_path, scan_num, det_Npixel, cen_x, cen_y, print_mode='debug'):
    """Load data from original HDF5 files"""
    det_xwidth = int(det_Npixel / 2)

    files = glob.glob(f"{base_path}/ptycho/{scan_num:03d}/*{scan_num:03d}_*.h5")
    N_lines = max(int(name.split("_")[-2]) for name in files)  # number of scan lines
    N_pts = max(int(name.split("_")[-1][:-3]) for name in files)  # number of scan points per line
    verbose_print(f"Number of scan lines: {N_lines}, Number of scan points per line: {N_pts}", print_mode)

    pos = []
    dp = []
    for line in range(N_lines):
        verbose_print(f"Loading scan line No.{line + 1}...", print_mode)

        start_time = time.time()
        for point in range(N_pts):
            pos_file = glob.glob(
                f"{base_path}/positions/{scan_num:03d}/*{scan_num:03d}_{line + 1:05d}_{point:d}.dat"
            )[0]
            pos_arr = np.genfromtxt(pos_file, delimiter="")
            pos.append(np.mean(pos_arr, axis=0))

            h5_file = glob.glob(
                f"{base_path}/ptycho/{scan_num:03d}/*{scan_num:03d}_{line + 1:05d}_{point:d}.h5"
            )[0]
            with h5py.File(h5_file, "r") as h5_data:
                filePath = "entry/data/data"
                dp_temp = h5_data[filePath][...]
                dp_temp[dp_temp < 0] = 0
                dp_temp[dp_temp > 1e6] = 0

                index_x_lb = int(cen_x - det_Npixel // 2)
                index_x_ub = int(cen_x + (det_Npixel + 1) // 2)
                index_y_lb = int(cen_y - det_Npixel // 2)
                index_y_ub = int(cen_y + (det_Npixel + 1) // 2)
                dp_crop = dp_temp[:, index_y_lb:index_y_ub, index_x_lb:index_x_ub]
                dp.append(dp_crop)

    positions = np.array(pos)

    dp = np.concatenate(dp, axis=0) if dp else np.array([])  # Concatenate if dp is not empty

    return dp, positions


def load_data_12idc_tiff(base_path, scan_num, det_Npixel, cen_x, cen_y, print_mode='debug'):
    """
    Load data from TIFF files and position files.
    This implements functionality similar to process_scan from the data_preprocess script.
    """
    import tifffile

    verbose_print("Loading data from TIFF files and position files.", print_mode)
    # Define paths
    tif_dir = os.path.join(base_path, "tifs", f"{scan_num:03d}")

    # Get all tif files for the scan
    tif_files = glob.glob(os.path.join(tif_dir, f"*_{scan_num:03d}_*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No tif files found for scan {scan_num}.")

    # Extract sample name from the first file
    sample_name = os.path.basename(tif_files[0]).split(f"_{scan_num:03d}_")[0]

    # Group files by line
    line_dict = {}
    for tif_file in tif_files:
        basename = os.path.basename(tif_file)
        parts = basename.split("_")
        line = int(parts[-2])  # Extract line number
        point = int(parts[-1].split(".")[0])  # Extract point number
        if line not in line_dict:
            line_dict[line] = []
        line_dict[line].append((point, tif_file))

    # Initialize lists to store all diffraction patterns and positions
    all_dps = []
    all_positions = []

    # Process each line
    for line, point_files in line_dict.items():
        # Sort by point number
        point_files.sort(key=lambda x: x[0])

        # Process each point
        for point, tif_path in point_files:
            # Load the tif file
            dp = tifffile.imread(tif_path)

            # Process the position file
            pos_file = os.path.join(
                base_path,
                "positions",
                f"{scan_num:03d}",
                f"{sample_name}_{scan_num:03d}_{line:05d}_{point - 1:d}.dat",
            )
            if os.path.exists(pos_file):
                pos_arr = np.genfromtxt(pos_file, delimiter="")
                avg_pos = np.mean(pos_arr, axis=0)
            else:
                verbose_print(f"Warning: Position file not found: {pos_file}", print_mode)
                avg_pos = np.array([np.nan, np.nan])

            # Crop the diffraction pattern to the requested size
            index_x_lb = int(cen_x - det_Npixel // 2)
            index_x_ub = int(cen_x + (det_Npixel + 1) // 2)
            index_y_lb = int(cen_y - det_Npixel // 2)
            index_y_ub = int(cen_y + (det_Npixel + 1) // 2)

            # Check if the diffraction pattern is large enough to crop
            if (
                dp.shape[0] >= index_y_ub
                and dp.shape[1] >= index_x_ub
                and index_y_lb >= 0
                and index_x_lb >= 0
            ):
                dp_cropped = dp[index_y_lb:index_y_ub, index_x_lb:index_x_ub]
                all_dps.append(dp_cropped)
                all_positions.append(avg_pos)
            else:
                verbose_print(f"Warning: Diffraction pattern too small to crop: {tif_path}", print_mode)

    # Stack all diffraction patterns and positions
    if not all_dps:
        raise ValueError("No valid diffraction patterns found after cropping")

    dp_stack = np.stack(all_dps)
    positions = np.array(all_positions)

    # Process positions: extract, invert x, center, and reshape
    positions_processed = np.zeros((len(positions), 2))
    positions_processed[:, 0] = positions[:, 1] * 1e-9 - np.mean(
        positions[:, 1] * 1e-9
    )  # y positions
    positions_processed[:, 1] = -positions[:, 2] * 1e-9 - np.mean(
        -positions[:, 2] * 1e-9
    )  # x positions

    dp_stack = np.clip(dp_stack, 0, 1e6)  # Replace both operations with a single clip

    return dp_stack, positions_processed


def load_data_bnp(base_path, scan_num, det_Npixel, cen_x, cen_y, print_mode='debug'):
    verbose_print("Loading scan positions and diffraction patterns measured by the Bionanoprobe instrument.", print_mode)

    import re
    match = re.search(r'(20\d{2})', base_path)
    year = match.group(1) if match else None
    
    if year >= '2025': #after APS-U
        XRFfile_path = f"{base_path}/mda/bnp_fly{scan_num:04d}.mda" 
        if not os.path.exists(XRFfile_path):
            raise FileNotFoundError(f"The XRF file path does not exist: {XRFfile_path}")
        from .pear_utils_aps import readMDA
        XRFfile = readMDA(XRFfile_path)
        y_pos=np.array(XRFfile[1].p[0].data)
        x_pos=np.array(XRFfile[2].p[0].data)[0]
    else: # before APS-U
        XRFfile_path = f"{base_path}/img.dat/bnp_fly{scan_num:04d}.mda.h5"
        if not os.path.exists(XRFfile_path):
            raise FileNotFoundError(f"The XRF file path does not exist: {XRFfile_path}")
        XRFfile = h5py.File(XRFfile_path)
        x_pos = XRFfile["MAPS/x_axis"][:]
        y_pos = XRFfile["MAPS/y_axis"][:]

    dp_dir = f"{base_path}/ptycho/"
    filePath = "/entry/data/data"

    x_pos -= x_pos.mean()
    y_pos -= y_pos.mean()
    x_pos *= 1e-6
    y_pos *= 1e-6

    # Load diffraction patterns
    index_x_lb, index_x_ub = int(cen_x - det_Npixel // 2), int(cen_x + (det_Npixel + 1) // 2)
    index_y_lb, index_y_ub = int(cen_y - det_Npixel // 2), int(cen_y + (det_Npixel + 1) // 2)

    N_scan_y, N_scan_x = y_pos.size, x_pos.size
    pattern = os.path.join(dp_dir, f"bnp_fly{scan_num:04d}_*.h5")
    file_list = sorted(glob.glob(pattern))
    verbose_print(f"Number of diffraction patterns files: {len(file_list)}", print_mode)

    if len(file_list) < N_scan_y:
        verbose_print(f"Only {len(file_list)} diffraction files found, adjusting N_scan_y from {N_scan_y} to {len(file_list)}.", print_mode)
        N_scan_y = len(file_list)
    
    verbose_print(f"N_scan_y={N_scan_y}, N_scan_x={N_scan_x}, N_scan_dp={N_scan_x * N_scan_y}", print_mode)

    dp, scan_posx, scan_posy = [], [], []

    for i in range(N_scan_y):
        dp_file_name =  f"bnp_fly{scan_num:04d}_{i:06d}.h5"
        fileName = os.path.join(dp_dir, dp_file_name)
        verbose_print(f"Loading scan line No.{i + 1} from {dp_file_name}", print_mode)

        with h5py.File(fileName, "r") as h5_data:
            # h5_data = h5py.File(fileName,'r')
            dp_temp = h5_data[filePath][...]
            dp_temp[dp_temp < 0] = 0
            dp_temp[dp_temp > 1e7] = 0
            # print(fileName, dp_temp.shape)
            # dp_temp = np.clip(h5_data[filePath][...], 0, 1e7)
            # dp_temp = np.clip(h5_data[filePath][...], 0, 1e7)
            if dp_temp.shape[0] < 5:
                verbose_print(f"A lot of pixels are missed on this line: {dp_temp.shape[0]} pixels, Skip!", print_mode)
                continue

            dp_crop = dp_temp[:, index_y_lb:index_y_ub, index_x_lb:index_x_ub]
            dp.append(dp_crop)
            scan_posx.extend(x_pos[: dp_crop.shape[0]])
            scan_posy.extend([y_pos[i]] * dp_crop.shape[0])

    positions = np.column_stack((scan_posy, scan_posx))
    dp = np.concatenate(dp, axis=0) if dp else np.array([])  # Concatenate if dp is not empty

    return dp, positions


def _read_lynx_position_file(posfile):
    """
    Read a position file and return a dictionary with header info and data columns.

    Expected file format:
      - The first line contains: <string> <integer>, <string> <float>
      - The second line contains column names separated by whitespace.
      - The remaining lines contain numeric data corresponding to the columns.

    Parameters:
        posfile (str): Path to the position file.

    Returns:
        dict: Dictionary with header keys and data arrays.
    """
    if not os.path.exists(posfile):
        raise FileNotFoundError(f"Position file {posfile} not found")

    struct_out = {}
    with open(posfile, "r") as f:
        # Read and parse the header line
        header_line = f.readline().strip()
        parts = header_line.split(",")
        if len(parts) != 2:
            raise ValueError("Header line format is incorrect")

        # Process first part (key and integer value)
        key1_val = parts[0].strip().split()
        if len(key1_val) != 2:
            raise ValueError("First header part format is incorrect")
        key1, val1 = key1_val[0], key1_val[1]

        # Process second part (key and float value)
        key2_val = parts[1].strip().split()
        if len(key2_val) != 2:
            raise ValueError("Second header part format is incorrect")
        key2, val2 = key2_val[0], key2_val[1]

        struct_out[key1] = int(val1)
        struct_out[key2] = float(val2)

        # Read column names (second line)
        names_line = f.readline().strip()
        names = names_line.split()

        # Read the remaining numerical data using numpy
        data = np.loadtxt(f)
        if data.ndim == 1:
            data = data.reshape(-1, len(names))

        # Assign each column to the dictionary using the column names
        for i, name in enumerate(names):
            struct_out[name] = data[:, i]

    return struct_out


def load_data_lynx(base_path, scan_num, det_Npixel, cen_x, cen_y, print_mode='debug'):
    verbose_print("Loading scan positions and diffraction patterns measured by the LYNX instrument.", print_mode)
    # Load positions from .dat file
    pos_file = f"{base_path}/data/scan_positions/scan_{scan_num:05d}.dat"
    out_orch = _read_lynx_position_file(pos_file)

    x_positions = -out_orch["Average_x_st_fzp"]
    y_positions = -out_orch["Average_y_st_fzp"]

    # Convert to meters if needed (adjust multiplier as needed)
    x_positions = x_positions * 1e-6  # Adjust this factor based on your data units
    y_positions = y_positions * 1e-6

    # Stack positions
    positions = np.column_stack((y_positions, x_positions))

    # Determine subfolder based on scan number
    subfolder_start = (scan_num // 1000) * 1000
    subfolder_end = subfolder_start + 999
    data_dir = os.path.join(
        base_path, "data", "eiger_4", f"S{subfolder_start:05d}-{subfolder_end:05d}", f"S{scan_num:05d}"
    )
    file_path = os.path.join(data_dir, f"run_{scan_num:05d}_000000000000.h5")
    # Validate inputs
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Calculate detector ROI indices
    N_det_x = 1614
    N_det_y = 1030

    N_dp_x_max = min(cen_x, N_det_x-cen_x) * 2
    N_dp_y_max = min(cen_y, N_det_y-cen_y) * 2

    # crop square diffraction patterns to ensure isotropic resolution
    N_dp_crop = min(N_dp_x_max, N_dp_y_max, det_Npixel)
    if N_dp_crop < det_Npixel:
        verbose_print(f"The maximum size can be cropped from raw diffraction patterns is {N_dp_crop}", print_mode)

    index_x_lb = int(cen_x - N_dp_crop // 2)
    index_x_ub = int(cen_x + (N_dp_crop + 1) // 2)
    index_y_lb = int(cen_y - N_dp_crop // 2)
    index_y_ub = int(cen_y + (N_dp_crop + 1) // 2)

    # Load diffraction patterns
    with h5py.File(file_path, "r") as h5_data:
        dp_temp = h5_data["entry/data/eiger_4"][:]
        N_scan_dp = dp_temp.shape[0]
        verbose_print(f"Number of diffraction patterns: {N_scan_dp}", print_mode)

        # Initialize output array
        dp = np.zeros((N_scan_dp, N_dp_crop, N_dp_crop))

        # Process each diffraction pattern
        for j in range(N_scan_dp):
            dp[j] = dp_temp[j, index_y_lb:index_y_ub, index_x_lb:index_x_ub]
            #scipy.ndimage.zoom(roi, [1, 1], output=dp[j], order=1)

    # Clean up data
    dp[dp < 0] = 0
    dp[dp > 1e7] = 0

    return dp, positions


def load_data_velo(base_path, scan_num, det_Npixel, cen_x, cen_y, print_mode='debug'):
    verbose_print("Loading scan positions and diffraction patterns measured by the Velociprobe instrument.", print_mode)

    dp_dir = f"{base_path}/ptycho/fly{scan_num:03d}/"

    # Load scan positions from original file
    getpos_path = os.path.join(base_path, "positions")
    s = glob.glob(getpos_path + "/fly{:03d}_0.txt".format(scan_num))
    pos = np.genfromtxt(s[0], delimiter=",")

    x, y = [], []
    for trigger in range(1, int(pos[:, 7].max() + 1)):
        st = np.argwhere(pos[:, 7] == trigger)
        x.append((pos[st[0], 1] + pos[st[-1], 1]) / 2.0)  # LI
        y.append(-(pos[st[0], 5] + pos[st[-1], 5]) / 2.0)  # no LI
    ppX = np.asarray(x) * 1e-9
    ppY = np.asarray(y) * 1e-9
    rot_ang = 0
    ppX *= np.cos(rot_ang / 180 * np.pi)
    N_scan_pos = ppX.size
    verbose_print(f"Number of scan positions: {N_scan_pos}", print_mode)

    # Load diffraction patterns
    N_dp_x_input = det_Npixel
    N_dp_y_input = det_Npixel
    index_x_lb = (cen_x - np.floor(N_dp_x_input / 2.0)).astype(int)
    index_x_ub = (cen_x + np.ceil(N_dp_x_input / 2.0)).astype(int)
    index_y_lb = (cen_y - np.floor(N_dp_y_input / 2.0)).astype(int)
    index_y_ub = (cen_y + np.ceil(N_dp_y_input / 2.0)).astype(int)

    # Determine N_scan_y
    list = os.listdir(dp_dir)
    N_scan_y = len(list) - 1 - 1 - 1
    verbose_print(f"N_scan_y={N_scan_y}", print_mode)

    # Determine N_scan_x
    filePath = "entry/data/data"
    fileName = f"{dp_dir}fly{scan_num:03d}_data_{1:06d}.h5"
    h5_data = h5py.File(fileName, "r")
    dp_temp = h5_data[filePath][...]
    N_scan_x = dp_temp.shape[0]
    verbose_print(f"N_scan_x={N_scan_x}", print_mode)

    N_scan_x_lb = 0
    N_scan_y_lb = 0
    N_scan_dp = N_scan_x * N_scan_y
    verbose_print(f"N_scan_dp={N_scan_dp}", print_mode)

    resampleFactor = 1
    resizeFactor = 1

    dp = np.zeros((N_scan_dp, int(N_dp_y_input * resizeFactor), int(N_dp_x_input * resizeFactor)))
    verbose_print(f"{dp.shape}", print_mode)

    for i in range(N_scan_y):
        fileName = (
            dp_dir + "fly" + "%03d" % (scan_num) + "_data_" + "%06d" % (i + 1 + N_scan_y_lb) + ".h5"
        )

        h5_data = h5py.File(fileName, "r", libver="latest")
        dp_temp = h5_data[filePath][...]
        for j in range(N_scan_x):
            index = i * N_scan_x + j
            scipy.ndimage.interpolation.zoom(
                dp_temp[j + N_scan_x_lb, index_y_lb:index_y_ub, index_x_lb:index_x_ub],
                [resizeFactor, resizeFactor],
                dp[index, :, :],
                1,
            )

    dp[dp < 0] = 0
    dp[dp > 1e7] = 0

    dp = dp[::resampleFactor, :, :]

    if N_scan_pos > N_scan_dp:  # if there are more positions than dp
        ppX = ppX[0:N_scan_dp]
        ppY = ppY[0:N_scan_dp]
    else:
        dp = dp[0:N_scan_pos, :, :]

    # Shift positions to center around (0,0)
    ppX = ppX - (np.max(ppX) + np.min(ppX)) / 2
    ppY = ppY - (np.max(ppY) + np.min(ppY)) / 2
    positions = np.column_stack((ppY, ppX))

    return dp, positions

