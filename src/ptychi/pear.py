import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import (set_default_complex_dtype,
                          generate_initial_opr_mode_weights)
import os
os.environ['HDF5_PLUGIN_PATH'] = '/mnt/micdata3/ptycho_tools/DectrisFileReader/HDF5Plugin'

from .pear_utils import select_gpu, generate_scan_list, FileBasedTracker, check_gpu_availability, verbose_print
from .pear_plot import plot_affine_evolution, plot_affine_summary
import numpy as np

import logging
#logging.basicConfig(level=logging.ERROR)
#logging.basicConfig(level=logging.INFO)
import time
from datetime import datetime  # Correct import for datetime.now()
import uuid
import json
import tempfile
import shutil
import fcntl
import gc

from .pear_io_aps import (initialize_recon,
                        save_reconstructions,
                        create_reconstruction_path,
                        save_initial_conditions)

# Global variable for print mode
print_mode = 'debug'

def ptycho_recon(run_recon=True, **params):
    global print_mode
    print_mode = params.get('print_mode', 'debug')
    if print_mode == 'prod':
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO)

    if params['gpu_id'] is None:
        params['gpu_id'] = select_gpu(params)
        print(f"Auto-selected GPU: {params['gpu_id']} for scan {params['scan_num']}")
    else:
        print(f"Using GPU: {params['gpu_id']} for scan {params['scan_num']}")

    # Set up computing device
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [params['gpu_id'] ]))

    import torch
    num_gpus = torch.cuda.device_count()
    
    if params['batch_selection_scheme'] != 'random' and num_gpus > 1:
        raise ValueError("Only 'random' batch selection scheme is currently supported for multiple GPUs.")

    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
 
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float32)
    set_default_complex_dtype(torch.complex64)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU device.")

    # Load data + preprocessing
    (dp, init_positions_px, init_probe, init_object, params) = initialize_recon(params)

    #recon parameters 
    options = api.LSQMLOptions()
    options.data_options.data = dp
    options.data_options.save_data_on_device = True if num_gpus == 1 else False
    
    options.data_options.wavelength_m = params['wavelength_m']
    #options.data_options.detector_pixel_size_m = det_pixel_size_m # Only useful for near-field ptycho
    
    options.object_options.initial_guess = init_object
    options.object_options.pixel_size_m = params['obj_pixel_size_m']
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.build_preconditioner_with_all_modes = False

    if params.get('object_smoothness_alpha', 0) > 0:
        options.object_options.smoothness_constraint.enabled = True
        options.object_options.smoothness_constraint.alpha = params['object_smoothness_alpha']

    if params.get('object_regularization_llm', False):
        options.object_options.regularization_llm.enabled = True

    # multislice parameters
    if params['object_thickness_m'] > 0 and params['number_of_slices'] > 1:
        params['slice_distance_m'] = params['object_thickness_m'] / params['number_of_slices']
        options.object_options.slice_spacings_m = [params['slice_distance_m']] * (params['number_of_slices'] - 1)
        options.object_options.optimal_step_size_scaler = 0.9
        options.object_options.multislice_regularization.enabled = params['layer_regularization'] > 0
        options.object_options.multislice_regularization.weight = params['layer_regularization']
        options.object_options.multislice_regularization.unwrap_phase = True
        options.object_options.multislice_regularization.unwrap_image_grad_method = api.enums.ImageGradientMethods.FOURIER_DIFFERENTIATION
        options.object_options.multislice_regularization.unwrap_image_integration_method = api.enums.ImageIntegrationMethods.FOURIER
        if params['position_correction_layer'] and params['position_correction']:
            options.probe_position_options.correction_options.slice_for_correction = params['position_correction_layer']
    
    options.object_options.step_size = 1
    options.object_options.multimodal_update = params['update_object_w_higher_probe_modes']
    options.object_options.patch_interpolation_method = api.PatchInterpolationMethods.FOURIER
    options.object_options.remove_object_probe_ambiguity = api.options.base.RemoveObjectProbeAmbiguityOptions(enabled=True)
    
    options.probe_options.initial_guess = init_probe
    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = 1
    options.probe_options.optimization_plan.start = params.get('probe_update_start_iteration', 1)
    #options.probe_options.optimization_plan.stop = None
    
    options.probe_options.orthogonalize_incoherent_modes.enabled = True    
    options.probe_options.orthogonalize_incoherent_modes.method = api.OrthogonalizationMethods.SVD
    options.probe_options.orthogonalize_opr_modes.enabled = True
    
    options.probe_options.center_constraint.enabled = params['center_probe']
    options.probe_options.support_constraint.enabled = params['probe_support']

    # position correction
    options.probe_position_options.position_x_px = init_positions_px[:, 1]
    options.probe_position_options.position_y_px = init_positions_px[:, 0]
    options.probe_position_options.optimizable = params['position_correction']
    options.probe_position_options.optimizer = api.Optimizers.SGD
    options.probe_position_options.step_size = 1
    options.probe_position_options.correction_options.correction_type = api.PositionCorrectionTypes.GRADIENT
    options.probe_position_options.optimization_plan.start = params.get('position_correction_start_iteration', 1)

    if params.get('position_correction_gradient_method', 'gaussian') == 'gaussian':
        options.probe_position_options.correction_options.differentiation_method = api.ImageGradientMethods.GAUSSIAN
    elif params.get('position_correction_gradient_method', 'gaussian') == 'fourier':
        options.probe_position_options.correction_options.differentiation_method = api.ImageGradientMethods.FOURIER_DIFFERENTIATION
    else:
        raise ValueError(f"Invalid position correction gradient method: {params['position_correction_gradient_method']}")
    options.probe_position_options.correction_options.update_magnitude_limit = params['position_correction_update_limit']
    options.probe_position_options.affine_transform_constraint.enabled = True # always calculate the affine matrix
    options.probe_position_options.affine_transform_constraint.apply_constraint = params['position_correction_affine_constraint']
    options.probe_position_options.affine_transform_constraint.position_weight_update_interval = 100 # TODO: add to params
    
    # variable probe correction
    #options.probe_position_options.correction_options.gradient_method = api.PositionCorrectionGradientMethods.FOURIER
    if params['number_opr_modes'] > 0:
        options.opr_mode_weight_options.initial_weights = generate_initial_opr_mode_weights(len(init_positions_px), init_probe.shape[0])
        options.opr_mode_weight_options.optimizable = True
        options.opr_mode_weight_options.update_relaxation = 0.1
        options.opr_mode_weight_options.smoothing.enabled = False
        options.opr_mode_weight_options.smoothing.method = api.OPRWeightSmoothingMethods.MEDIAN
        options.opr_mode_weight_options.smoothing.polynomial_degree = 4

    options.opr_mode_weight_options.optimize_intensity_variation = params['intensity_correction']

    # convergence parameters
    # Set batch size based on parameters
    N_dp = dp.shape[0]
    params['number_of_diffraction_patterns'] = N_dp
    if params['update_batch_size'] is not None:
        options.reconstructor_options.batch_size = params['update_batch_size']
        params['number_of_batches'] = N_dp // options.reconstructor_options.batch_size
        verbose_print(f"User-specified batch size: {params['update_batch_size']} " 
                      f"({params['number_of_batches']} batches for {N_dp} data points)", 
                      print_mode)
    elif params['number_of_batches'] is not None:
        # Calculate batch size from number of batches
        params['update_batch_size'] = max(1, N_dp // params['number_of_batches'])
        options.reconstructor_options.batch_size = params['update_batch_size']
       
        verbose_print(f"User-specified batch size: {params['update_batch_size']} " 
                      f"({params['number_of_batches']} batches for {N_dp} data points)", 
                      print_mode)
    else:
        #params['auto_batch_size_adjustment'] = True
        # Auto-configure based on batch selection scheme
        # Use smaller number of batches for 'compact' scheme
        params['number_of_batches'] = 1 if params['batch_selection_scheme'] == 'compact' else 10
        params['update_batch_size'] = max(1, N_dp // params['number_of_batches'])
        options.reconstructor_options.batch_size = params['update_batch_size']
        
        # Log the auto-configuration for transparency
        verbose_print(f"Auto-configured batch size: {params['update_batch_size']} " 
                      f"({params['number_of_batches']} batches for {N_dp} data points)", 
                      print_mode)

    #options.reconstructor_options.forward_model_options.pad_for_shift = 16
    #options.reconstructor_options.use_low_memory_forward_model = True
    if params['batch_selection_scheme'] == 'random':
        options.reconstructor_options.batching_mode = api.BatchingModes.RANDOM
    elif params['batch_selection_scheme'] == 'uniform':
        options.reconstructor_options.batching_mode = api.BatchingModes.UNIFORM
    elif params['batch_selection_scheme'] == 'compact':
        options.reconstructor_options.batching_mode = api.BatchingModes.COMPACT
        options.reconstructor_options.compact_mode_update_clustering = False
    if params['momentum_acceleration']:
        options.reconstructor_options.momentum_acceleration_gain = 0.5
        options.reconstructor_options.momentum_acceleration_gradient_mixing_factor = 1

    options.reconstructor_options.solve_step_sizes_only_using_first_probe_mode = True
    
    if params.get('noise_model', 'gaussian') == 'poisson':
        options.reconstructor_options.noise_model = api.NoiseModels.POISSON
    else: 
        options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
    
    if params.get("near_field_ptycho", False):
        options.data_options.free_space_propagation_distance_m = params['det_sample_dist_m']
        options.data_options.fft_shift = False
        options.reconstructor_options.forward_model_options.pad_for_shift = 50 # increase if fringe artifacts are present

    options.reconstructor_options.num_epochs = params['number_of_iterations']
    options.reconstructor_options.use_double_precision_for_fft = False
    options.reconstructor_options.default_dtype = api.Dtypes.FLOAT32

    options.reconstructor_options.allow_nondeterministic_algorithms = True # a bit faster

    options.reconstructor_options.forward_model_options.diffraction_pattern_blur_sigma = params.get('diffraction_pattern_blur', None)

    recon_path = create_reconstruction_path(params, options)
    save_initial_conditions(recon_path, params, options)

    task = PtychographyTask(options)
    
    if not run_recon:
        return task, recon_path, params
    
    for i in range(params['number_of_iterations'] // params['save_freq_iterations']):
        task.run(params['save_freq_iterations'])
        save_reconstructions(task, recon_path, params['save_freq_iterations']*(i+1), params)

    return task, recon_path, params
    

def ptycho_batch_recon(base_params):
    """
    Reconstruct a range of ptychography scans with automatic error handling and status tracking.
    
    Args:
        base_params: Dictionary of parameters to use as a template for all scans
            start_scan: First scan number to reconstruct
            end_scan: Last scan number to reconstruct (inclusive)
            scan_list: List of scan numbers to reconstruct. Will override start_scan, end_scan and scan_order.
            log_dir_suffix: Optional suffix for the log directory
            scan_order: Order to reconstruct the scans ('ascending', 'descending', or 'random')
            exclude_scans: List of scan numbers to exclude from reconstruction
            overwrite_ongoing: Whether to overwrite scans marked as ongoing
            overwrite_ongoing_min_age_hour: Minimum age of ongoing scans to overwrite (in hours)
            reset_scan_list: Whether to reset the scan list and reconstruct all scans again
            skip_error_types: List of error types to skip.
            
    The function creates a tracker to monitor the status of each scan and processes
    them according to the specified order, skipping completed scans unless forced to reconstruct again.
    """
    # Extract parameters from base_params
    start_scan = base_params.get('start_scan')
    end_scan = base_params.get('end_scan')
    log_dir_suffix = base_params.get('log_dir_suffix', '')
    scan_order = base_params.get('scan_order', 'ascending')
    exclude_scans = base_params.get('exclude_scans', [])
    overwrite_ongoing = base_params.get('overwrite_ongoing', False)
    overwrite_ongoing_min_age_hour = base_params.get('overwrite_ongoing_min_age_hour', 0)
    reset_scan_list = base_params.get('reset_scan_list', False)
    wait_time_seconds = base_params.get('wait_time_seconds', 5)
    num_repeats = base_params.get('num_repeats', np.inf)
    auto_batch_size_adjustment = base_params.get('auto_batch_size_adjustment', False)
    skip_error_types = base_params.get('skip_error_types', '')

    log_dir = os.path.join(base_params['data_directory'], 'ptychi_recons', 
                          base_params['recon_parent_dir'], 
                          f'recon_logs_{log_dir_suffix}' if log_dir_suffix else 'recon_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create tracker
    tracker = FileBasedTracker(log_dir, overwrite_ongoing=overwrite_ongoing, overwrite_ongoing_min_age_hour=overwrite_ongoing_min_age_hour)
    
    # Generate a unique worker ID
    worker_id = f"worker_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    
    repeat_count = 0

    if base_params.get('scan_list', []) != []:
        scan_list = base_params.get('scan_list')
    else:
        scan_list = generate_scan_list(start_scan, end_scan, scan_order, exclude_scans)

    while repeat_count < num_repeats:
        successful_scans = []
        failed_scans = []
        ongoing_scans = []
    
        for scan_num in scan_list:
            # Create a copy of the parameters for this scan
            scan_params = base_params.copy()
            
            scan_params['scan_num'] = scan_num
            
            # Check status using tracker
            status = tracker.get_status(scan_num)
            
            if status == 'done':
                #print(f"Scan {scan_num} already completed, skipping reconstruction")
                successful_scans.append(scan_num)
                continue
                
            if status == 'ongoing' and not overwrite_ongoing:
                print(f"Scan {scan_num} already ongoing, skipping reconstruction")
                ongoing_scans.append(scan_num)
                continue

            # Check if previous reconstruction failed due to OOM
            prev_status = tracker.get_full_status(scan_num)

            if (
                prev_status
                and prev_status.get('error_type')
                and prev_status.get('error_type') in skip_error_types
            ):
                print(f"Scan {scan_num} previously failed with error type {prev_status.get('error_type')}, skipping")
                failed_scans.append((scan_num, prev_status.get('error')))
                continue

            if (
                auto_batch_size_adjustment
                and prev_status
                and prev_status.get('status') == 'failed'
                and prev_status.get('error_type') == 'out_of_memory'
            ):
                # Previous attempt failed due to OOM, increase batch count
                prev_num_batches = prev_status.get('number_of_batches')
                prev_batch_size = prev_status.get('number_of_batches')
                curr_num_batches = scan_params.get('number_of_batches')
                if (
                    prev_num_batches is not None
                    and curr_num_batches is not None
                    and prev_num_batches >= curr_num_batches
                ):
                    new_batches = prev_num_batches + 1
                    #TODO: find a good way to get number of diffraction patterns or batch size from previous reconstruction
                    #while (Ndp // new_batches) >= (Ndp // prev_num_batches):
                    #new_batches += 1
                    scan_params['number_of_batches'] = new_batches
                    print(
                        f"\033[93mScan {scan_num} previously failed with CUDA OOM. "
                        f"Increasing the number of batches from {prev_num_batches} to {scan_params['number_of_batches']}\033[0m"
                    )

            # Try to start reconstruction
            if not tracker.start_recon(scan_num, worker_id, scan_params):
                #print(f"Could not acquire lock for scan {scan_num}, skipping")
                continue
                
            print(f"\033[91mStarting reconstruction for scan {scan_num}\033[0m")
            start_time = time.time()
            
            try:
                # Run reconstruction as a subprocess
                import subprocess
                import sys
                import json
                
                # Create a directory for temp files
                temp_dir = os.path.join(base_params['data_directory'], 'ptychi_recons', 'temp_files')
                os.makedirs(temp_dir, exist_ok=True)
                
                # Create paths for temp files with scan number included
                params_path = os.path.join(temp_dir, f"scan_{scan_num:04d}_params.json")
                script_path = os.path.join(temp_dir, f"scan_{scan_num:04d}_script.py")
                
                # Save parameters to the JSON file
                with open(params_path, 'w') as params_file:
                    # Convert NumPy arrays to lists to make them JSON serializable
                    json_compatible_params = {}
                    for key, value in scan_params.items():
                        if isinstance(value, np.ndarray):
                            json_compatible_params[key] = value.tolist()
                        else:
                            json_compatible_params[key] = value
                    
                    json.dump(json_compatible_params, params_file, indent=2)
                
                try:
                    # Create a Python script for subprocess
                    script_content = f"""
import json
import sys
import os
from pathlib import Path

# Add the parent directory to path so imports work correctly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ptychi.pear import ptycho_recon

# Load parameters 
with open('{params_path}', 'r') as f:
    params = json.load(f)

# Run reconstruction with real-time output
ptycho_recon(run_recon=True, **params)
"""
                    # Write the script to the file
                    with open(script_path, 'w') as script_file:
                        script_file.write(script_content)
                    
                    # Run the script as a subprocess with output streamed in real-time
                    # Use unbuffered Python to ensure tqdm displays properly
                    process = subprocess.Popen(
                        [sys.executable, "-u", script_path],  # -u flag for unbuffered output
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=0,  # Unbuffered
                    )
                    
                    # Collect all output for error reporting
                    stdout_lines = []
                    stderr_lines = []
                    
                    # Use threading to read both stdout and stderr simultaneously
                    import threading
                    import queue
                    
                    stdout_queue = queue.Queue()
                    stderr_queue = queue.Queue()
                    
                    def read_stdout():
                        for line in iter(process.stdout.readline, ''):
                            stdout_queue.put(line)
                        stdout_queue.put(None)  # Signal end
                    
                    def read_stderr():
                        for line in iter(process.stderr.readline, ''):
                            stderr_queue.put(line)
                        stderr_queue.put(None)  # Signal end
                    
                    # Start threads to read stdout and stderr
                    stdout_thread = threading.Thread(target=read_stdout)
                    stderr_thread = threading.Thread(target=read_stderr)
                    stdout_thread.start()
                    stderr_thread.start()
                    
                    # Process output in real-time
                    stdout_done = False
                    stderr_done = False
                    
                    while not (stdout_done and stderr_done):
                        # Process stdout
                        try:
                            line = stdout_queue.get_nowait()
                            if line is None:
                                stdout_done = True
                            else:
                                stdout_lines.append(line)
                                if scan_params['gpu_id'] is not None:
                                    print(f"[S{scan_num:04d}-GPU{scan_params['gpu_id']}]{line}", end='')
                                else:
                                    print(f"[S{scan_num:04d}-GPU:Auto]{line}", end='')
                        except queue.Empty:
                            pass
                        
                        # Process stderr (tqdm output)
                        try:
                            line = stderr_queue.get_nowait()
                            if line is None:
                                stderr_done = True
                            else:
                                stderr_lines.append(line)
                                # Print stderr directly to terminal for tqdm
                                print(line, end='')
                        except queue.Empty:
                            pass
                        
                        # Small sleep to prevent busy waiting
                        time.sleep(0.01)
                    
                    # Wait for threads to complete
                    stdout_thread.join()
                    stderr_thread.join()
                    
                    # Wait for process to complete and get return code
                    return_code = process.wait()
                    
                    if return_code != 0:
                        # Create a custom exception with full output
                        full_stdout = ''.join(stdout_lines)
                        full_stderr = ''.join(stderr_lines)
                        error_output = f"STDOUT:\n{full_stdout}\nSTDERR:\n{full_stderr}"
                        raise subprocess.CalledProcessError(return_code, f"{sys.executable} {script_path}", error_output)
                    
                    # If we reached here, reconstruction was successful
                    elapsed_time = time.time() - start_time
                    print(f"Scan {scan_num} completed successfully in {elapsed_time:.2f} seconds")
                    successful_scans.append(scan_num)
                    
                    # Update status with batch information
                    tracker.complete_recon(
                        scan_num, 
                        success=True,
                        number_of_batches=scan_params.get('number_of_batches')
                    )
                    
                except subprocess.CalledProcessError as e:
                    # Handle subprocess failure
                    elapsed_time = time.time() - start_time
                    error_message = f"Subprocess failed with exit code {e.returncode}"
                    
                    # Capture the full exception message, including output if any
                    if hasattr(e, 'output') and e.output:
                        error_detail = f"{error_message}\n\nFull script output:\n{e.output}"
                    else:
                        error_detail = f"{error_message}\nException details:\n{str(e)}"
                    
                    print(f"Scan {scan_num} failed after {elapsed_time:.2f} seconds")
                    print(f"Error details:\n{error_detail}")
                    failed_scans.append((scan_num, error_detail))
                    
                    # Update status with batch information
                    tracker.complete_recon(
                        scan_num, 
                        success=False, 
                        error=error_detail,
                        number_of_batches=scan_params.get('number_of_batches')
                    )
                finally:
                    # Optionally remove the temporary files when done
                    # Uncomment these lines if you want to clean up after successful runs
                    if os.path.exists(params_path):
                        os.unlink(params_path)
                    if os.path.exists(script_path):
                        os.unlink(script_path)
                    
                    # Give system time to fully clean up resources
                    print(f"Waiting for {wait_time_seconds} seconds before next scan...")
                    time.sleep(wait_time_seconds)

                if reset_scan_list:
                    # Break the for loop after processing the current scan
                    break
                    
            except Exception as e:
                # Handle failure
                elapsed_time = time.time() - start_time
                print(f"Scan {scan_num} failed after {elapsed_time:.2f} seconds with error: {str(e)}")
                failed_scans.append((scan_num, str(e)))
                
                # Update status with batch information
                tracker.complete_recon(
                    scan_num, 
                    success=False, 
                    error=str(e),
                    number_of_batches=scan_params.get('number_of_batches')
                )
        
        # Print summary of processing
        #print(f"Successfully processed scans: {successful_scans}")
        print(f"Number of completed scans:{len(successful_scans)}/{len(scan_list)}.")
        print(f"Number of failed scans:{len(failed_scans)}.")
        print(f"Number of ongoing scans:{len(ongoing_scans)}.")
        print(f"Number of excluded scans: {len(set(scan_list) & set(exclude_scans))}")
        
        if len(successful_scans) == len(scan_list) - len(exclude_scans):
            print(f"All scans completed successfully")
            break
        else:
            repeat_count += 1
            print(f"Waiting for {wait_time_seconds} seconds...")
            time.sleep(wait_time_seconds)
   
    print(f"Batch processing complete.")

def ptycho_batch_recon_affine_calibration(base_params):
    """
    Automatically calibrate the geometric parameters based on coarse reconstructions.
    
    Args:
        base_params: Dictionary of parameters to use as a template for all scans
            start_scan: First scan number to process
            end_scan: Last scan number to process (inclusive)
    """
    import subprocess
    import sys
    import json
    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    N_runs = 1
    
    # Extract parameters
    start_scan = base_params.get('start_scan')
    end_scan = base_params.get('end_scan')
    scan_list = list(range(start_scan, end_scan + 1))
    det_sample_dist_m = base_params['det_sample_dist_m']  # initial distance
    
    # Setup directories
    geom_calibration_dir = os.path.join(base_params['data_directory'], 'ptychi_recons', 
                                        base_params['recon_parent_dir'], 'geom_calibration')
    temp_dir = os.path.join(base_params['data_directory'], 'ptychi_recons', 'temp_files')
    os.makedirs(geom_calibration_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Parameters to track and plot
    params_to_plot = ['scale', 'asymmetry', 'rotation', 'shear']
    affine_params = {}
    
    for i in range(N_runs):
        print(f"\033[94mCalibration run {i+1}/{N_runs} with distance {det_sample_dist_m}m\033[0m")
        
        for scan_num in scan_list:
            # Create scan-specific parameters
            scan_params = base_params.copy()
            scan_params['scan_num'] = scan_num
            scan_params['det_sample_dist_m'] = det_sample_dist_m
            scan_params['recon_dir_suffix'] = f'd{det_sample_dist_m}'
            
            print(f"\033[91mStarting reconstruction for scan {scan_num}\033[0m")
            
            # Create temporary files
            params_path = os.path.join(temp_dir, f"scan_{scan_num:04d}_params.json")
            script_path = os.path.join(temp_dir, f"scan_{scan_num:04d}_script.py")
            
            # Save parameters to JSON
            with open(params_path, 'w') as params_file:
                json_compatible_params = {
                    key: value.tolist() if isinstance(value, np.ndarray) else value 
                    for key, value in scan_params.items()
                }
                json.dump(json_compatible_params, params_file, indent=2)
            
            # Create Python script for subprocess
            script_content = f"""
import json
import sys
import os
from pathlib import Path

# Add the parent directory to path so imports work correctly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ptychi.pear import ptycho_recon

# Load parameters 
with open('{params_path}', 'r') as f:
    params = json.load(f)

# Run reconstruction with real-time output
ptycho_recon(run_recon=True, **params)
"""
            with open(script_path, 'w') as script_file:
                script_file.write(script_content)
            
            # Run subprocess with real-time output
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            
            recon_path = None
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
                if "Reconstruction results will be saved in:" in line:
                    recon_path = line.split("Reconstruction results will be saved in:")[1].strip()
            
            # Wait for process to complete
            return_code = process.wait()
            
            # Clean up temporary files
            for path in [params_path, script_path]:
                if os.path.exists(path):
                    os.unlink(path)
            
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, f"{sys.executable} {script_path}")    
            
            # Extract affine parameters from reconstruction file
            try:
                h5_path = f'{recon_path}/recon_Niter{base_params["number_of_iterations"]}.h5'
                with h5py.File(h5_path, 'r') as f:
                    if 'pos_corr' in f:
                        affine_params_temp = {}
                        for var in params_to_plot + ['iterations']:
                            if var in f['pos_corr']:
                                affine_params_temp[var] = f[f'/pos_corr/{var}'][:]
                        
                        if affine_params_temp:
                            affine_params[scan_num] = affine_params_temp
                            print(f"Scan {scan_num}: Position correction data loaded")
                        else:
                            print(f"Scan {scan_num}: Position correction data not found in file")
            except Exception as e:
                print(f"Error reading HDF5 file for scan {scan_num}: {str(e)}")
        
        print(f"Batch processing complete for run {i+1}.")
        
        # Plot parameter evolution
        fig_path = os.path.join(geom_calibration_dir, f'affine_evolution_d{det_sample_dist_m}.png')
        plot_affine_evolution(affine_params, params_to_plot, fig_path)
        
        # Plot final parameter values
        fig_path = os.path.join(geom_calibration_dir, f'affine_summary_d{det_sample_dist_m}.png')
        plot_affine_summary(affine_params, params_to_plot, fig_path)
        
        # Save calibration results
        calibration_results = {
            'params': {},
            'calibrated_distance': det_sample_dist_m
        }
        
        for param in params_to_plot:
            final_values = [data[param][-1] for data in affine_params.values() if param in data]
            if final_values:
                calibration_results['params'][param] = {
                    'mean': float(np.mean(final_values)),
                    'std': float(np.std(final_values)),
                    'min': float(np.min(final_values)),
                    'max': float(np.max(final_values)),
                    'values': {scan_num: float(data[param][-1]) 
                              for scan_num, data in affine_params.items() if param in data}
                }
        
        # Calculate calibrated distance for next run
        old_distance = det_sample_dist_m
        if 'scale' in calibration_results['params']:
            mean_scale = calibration_results['params']['scale']['mean']
            calibrated_distance = round(det_sample_dist_m / mean_scale, 4)
            calibration_results['calibrated_distance'] = calibrated_distance
            det_sample_dist_m = calibrated_distance
            print(f"Calibrated distance: {det_sample_dist_m}m")
        
        # Save to file
        calibration_file = os.path.join(geom_calibration_dir, f'calibration_results_d{old_distance}.json')
        with open(calibration_file, 'w') as f:
            json.dump(calibration_results, f, indent=4)
        print(f"Saved calibration results to: {calibration_file}")

def ptycho_batch_recon_affine_calibration2(base_params):
    """
    Automatically calibrate the geometric parameters based on coarse reconstructions.
    Run multiple reconstructions in parallel using different GPUs.
    
    Args:
        base_params: Dictionary containing:
            - gpu_ids: List of GPU IDs to use for parallel processing
            - start_scan, end_scan: Range of scans to process
            - det_sample_dist_m: Initial detector-sample distance
            - Other standard reconstruction parameters
    """
    import subprocess
    import sys
    import json
    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from concurrent.futures import ThreadPoolExecutor
    
    # Get available GPU IDs
    gpu_ids = base_params.get('gpu_id', [0])  # Default to GPU 0 if not specified
    max_workers = len(gpu_ids)  # Number of parallel processes = number of GPUs
    print(f"Using {max_workers} GPUs: {gpu_ids}")
    
    def run_single_reconstruction(scan_num, gpu_id, det_sample_dist_m):
        """Run a single reconstruction on specified GPU"""
        # Create scan-specific parameters
        scan_params = base_params.copy()
        scan_params['scan_num'] = scan_num
        scan_params['det_sample_dist_m'] = det_sample_dist_m
        scan_params['recon_dir_suffix'] = f'd{det_sample_dist_m}'
        scan_params['gpu_id'] = gpu_id  # Specify a single GPU to use
        
        print(f"\033[91mStarting reconstruction for scan {scan_num} on GPU {gpu_id}\033[0m")
        
        # Create temporary files
        params_path = os.path.join(temp_dir, f"scan_{scan_num:04d}_gpu{gpu_id}_params.json")
        script_path = os.path.join(temp_dir, f"scan_{scan_num:04d}_gpu{gpu_id}_script.py")
        
        try:
            # Save parameters to JSON
            with open(params_path, 'w') as params_file:
                json_compatible_params = {
                    key: value.tolist() if isinstance(value, np.ndarray) else value 
                    for key, value in scan_params.items()
                }
                json.dump(json_compatible_params, params_file, indent=2)
            
            # Create Python script for subprocess
            script_content = f"""
import json
import sys
import os
import numpy as np
from pathlib import Path

# Add the parent directory to path so imports work correctly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ptychi.pear import ptycho_recon

# Load parameters 
with open('{params_path}', 'r') as f:
    params = json.load(f)

# Convert lists back to NumPy arrays where needed
for key, value in params.items():
    if isinstance(value, list) and key in ['scan_positions', 'positions']:
        params[key] = np.array(value)

# Run reconstruction
ptycho_recon(run_recon=True, **params)
"""
            with open(script_path, 'w') as script_file:
                script_file.write(script_content)
            
            # Run subprocess
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            
            recon_path = None
            last_print_time = time.time()
            output_buffer = []
            
            # Stream output in real-time with GPU identifier
            for line in iter(process.stdout.readline, ''):
                output_buffer.append(line)
                if "Reconstruction results will be saved in:" in line:
                    recon_path = line.split("Reconstruction results will be saved in:")[1].strip()
                
                # Print the most recent line every 5 seconds
                current_time = time.time()
                if current_time - last_print_time > 5:
                    if output_buffer:
                        print(f"[Scan {scan_num}, GPU {gpu_id}] {output_buffer[-1]}", end='')
                    output_buffer = []
                    last_print_time = current_time
            
            # Print any remaining output
            # if output_buffer:
            #     print(f"[Scan {scan_num}, GPU {gpu_id}] {output_buffer[-1]}", end='')
            
            # Wait for process to complete
            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, f"{sys.executable} {script_path}")
                
            return recon_path, scan_num
            
        finally:
            # Clean up temporary files
            for path in [params_path, script_path]:
                if os.path.exists(path):
                    os.unlink(path)
            
            # Ensure GPU memory is cleaned up
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    N_runs = 4
    params_to_plot = ['scale', 'asymmetry', 'rotation', 'shear']
    affine_params = {}
    
    # Setup directories
    geom_calibration_dir = os.path.join(base_params['data_directory'], 'ptychi_recons', 
                                       base_params['recon_parent_dir'], 'geom_calibration')
    temp_dir = os.path.join(base_params['data_directory'], 'ptychi_recons', 'temp_files')
    os.makedirs(geom_calibration_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Extract parameters
    start_scan = base_params.get('start_scan')
    end_scan = base_params.get('end_scan')
    scan_list = list(range(start_scan, end_scan + 1))
    det_sample_dist_m = base_params['det_sample_dist_m']
    
    for i in range(N_runs):
        print(f"\033[94mCalibration run {i+1}/{N_runs} with distance {det_sample_dist_m}m\033[0m")
        
        # Run reconstructions in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for each scan, cycling through available GPUs
            futures = []
            for idx, scan_num in enumerate(scan_list):
                gpu_id = gpu_ids[idx % len(gpu_ids)]
                future = executor.submit(run_single_reconstruction, scan_num, gpu_id, det_sample_dist_m)
                futures.append((future, scan_num))
            
            # Collect results as they complete
            for future, scan_num in futures:
                try:
                    recon_path, _ = future.result()
                    
                    # Extract affine parameters from reconstruction file
                    try:
                        h5_path = f'{recon_path}/recon_Niter{base_params["number_of_iterations"]}.h5'
                        with h5py.File(h5_path, 'r') as f:
                            if 'pos_corr' in f:
                                affine_params_temp = {}
                                for var in params_to_plot + ['iterations']:
                                    if var in f['pos_corr']:
                                        affine_params_temp[var] = f[f'/pos_corr/{var}'][:]
                                
                                if affine_params_temp:
                                    affine_params[scan_num] = affine_params_temp
                                    print(f"Scan {scan_num}: Position correction data loaded")
                    except Exception as e:
                        print(f"Error reading HDF5 file for scan {scan_num}: {str(e)}")
                        
                except Exception as e:
                    print(f"Error processing scan {scan_num}: {str(e)}")
        
        print(f"Batch processing complete for run {i+1}.")
        
        # Plot parameter evolution
        fig_path = os.path.join(geom_calibration_dir, f'affine_evolution_d{det_sample_dist_m}.png')
        plot_affine_evolution(affine_params, params_to_plot, fig_path)
        
        # Plot final parameter values
        fig_path = os.path.join(geom_calibration_dir, f'affine_summary_d{det_sample_dist_m}.png')
        plot_affine_summary(affine_params, params_to_plot, fig_path)
        
        # Save calibration results
        calibration_results = {
            'params': {},
            'calibrated_distance': det_sample_dist_m
        }
        
        for param in params_to_plot:
            final_values = [data[param][-1] for data in affine_params.values() if param in data]
            if final_values:
                calibration_results['params'][param] = {
                    'mean': float(np.mean(final_values)),
                    'std': float(np.std(final_values)),
                    'min': float(np.min(final_values)),
                    'max': float(np.max(final_values)),
                    'values': {scan_num: float(data[param][-1]) 
                              for scan_num, data in affine_params.items() if param in data}
                }
        
        # Calculate calibrated distance for next run
        old_distance = det_sample_dist_m
        if 'scale' in calibration_results['params']:
            mean_scale = calibration_results['params']['scale']['mean']
            calibrated_distance = round(det_sample_dist_m / mean_scale, 4)
            calibration_results['calibrated_distance'] = calibrated_distance
            det_sample_dist_m = calibrated_distance
            print(f"Calibrated distance: {det_sample_dist_m}m")
        
        # Save to file
        calibration_file = os.path.join(geom_calibration_dir, f'calibration_results_d{old_distance}.json')
        with open(calibration_file, 'w') as f:
            json.dump(calibration_results, f, indent=4)
        print(f"Saved calibration results to: {calibration_file}")

def print_summary(successful_scans, failed_scans, ongoing_scans, start_scan, end_scan):
    """Print a summary of the reconstruction results."""
    total_scans = end_scan - start_scan + 1
    
    print(f"\nProcessing complete:")
    print(f"Successfully processed scans: {successful_scans}")
    print(f"Number of completed scans: {len(successful_scans)}/{total_scans}")
    print(f"Number of failed scans: {len(failed_scans)}")
    print(f"Number of ongoing scans: {len(ongoing_scans)}")
    
    # if failed_scans:
    #     print("\nFailed scans and errors:")
    #     for scan_num, error in failed_scans:
    #         print(f"Scan {scan_num}: {error}")
