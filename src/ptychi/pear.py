import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import (set_default_complex_dtype,
                          generate_initial_opr_mode_weights)
import os
os.environ['HDF5_PLUGIN_PATH'] = '/mnt/micdata3/ptycho_tools/DectrisFileReader/HDF5Plugin'

from .pear_utils import select_gpu, generate_scan_list, FileBasedTracker
from .pear_plot import plot_affine_evolution, plot_affine_summary
import numpy as np

import logging
logging.basicConfig(level=logging.ERROR)
import time
from datetime import datetime  # Correct import for datetime.now()
import uuid
import json
import tempfile
import shutil
import fcntl

from .pear_io_aps import (initialize_recon,
                        save_reconstructions,
                        create_reconstruction_path,
                        save_initial_conditions)

def ptycho_recon(run_recon=True, **params):

    if params['gpu_id'] is None:
        params['gpu_id'] = select_gpu(params)
        print(f"Auto-selected GPU: {params['gpu_id']}")
    else:
        print(f"Using GPU: {params['gpu_id']}")

    # Set up computing device
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [params['gpu_id'] ]))
    import torch
    
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
    options.data_options.save_data_on_device = True
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
    if params['update_batch_size'] is not None:
        params['auto_batch_size_estimation'] = False
        options.reconstructor_options.batch_size = params['update_batch_size']
        params['number_of_batches'] = dp.shape[0] // options.reconstructor_options.batch_size
        print(f"User-specified batch size: {options.reconstructor_options.batch_size} " 
              f"({params['number_of_batches']} batches for {dp.shape[0]} data points)")
    elif params['number_of_batches'] is not None:
        params['auto_batch_size_estimation'] = False
        # Calculate batch size from number of batches
        total_data_points = dp.shape[0]
        options.reconstructor_options.batch_size = max(1, total_data_points // params['number_of_batches'])
        print(f"User-specified batch size: {options.reconstructor_options.batch_size} " 
              f"({params['number_of_batches']} batches for {dp.shape[0]} data points)")
    else:
        params['auto_batch_size_estimation'] = True
        # Auto-configure based on batch selection scheme
        total_data_points = dp.shape[0]
        # Use smaller number of batches for 'compact' scheme
        params['number_of_batches'] = 1 if params['batch_selection_scheme'] == 'compact' else 10
        options.reconstructor_options.batch_size = max(1, total_data_points // params['number_of_batches'])
        
        # Log the auto-configuration for transparency
        print(f"Auto-configured batch size: {options.reconstructor_options.batch_size} " 
              f"({params['number_of_batches']} batches for {total_data_points} data points)")

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
    
    options.reconstructor_options.num_epochs = params['number_of_iterations']
    options.reconstructor_options.use_double_precision_for_fft = False
    options.reconstructor_options.default_dtype = api.Dtypes.FLOAT32

    options.reconstructor_options.allow_nondeterministic_algorithms = True # a bit faster

    recon_path = create_reconstruction_path(params, options)
    save_initial_conditions(recon_path, params, options)

    task = PtychographyTask(options)
    
    if not run_recon:
        return task, recon_path, params
    
    try:
        if params['auto_batch_size_estimation']:
            # Set up a loop to handle potential out of memory errors
            max_retries = 10  # Limit the number of retries to prevent infinite loops
            retry_count = 0
            
            while retry_count <= max_retries:
                try:
                    # Try to run the reconstruction
                    for i in range(params['number_of_iterations'] // params['save_freq_iterations']):
                        task.run(params['save_freq_iterations'])
                        save_reconstructions(task, recon_path, params['save_freq_iterations']*(i+1), params)
                    break  # If successful, exit the retry loop
                    
                except RuntimeError as e:
                    error_msg = str(e)
                    # Check if this is an out of memory error
                    if "CUDA out of memory" in error_msg or "cudaErrorOutOfMemory" in error_msg:
                        retry_count += 1
                        print(f"CUDA out of memory error. Increasing number of batches by 1")
                        if retry_count > max_retries:
                            print(f"Failed after {max_retries} attempts. Giving up.")
                            raise  # Re-raise the exception if we've exceeded max retries
                        
                        # Update number of batches for logging
                        params['number_of_batches'] = params['number_of_batches'] + 1
                        options.reconstructor_options.batch_size = max(1, total_data_points // params['number_of_batches'])
                        print(f"CUDA out of memory error. Attempt {retry_count}/{max_retries}: "
                                f"Increasing to {params['number_of_batches']} batches")
                        print(f"New batch size: {options.reconstructor_options.batch_size} "
                                f"({params['number_of_batches']} batches for {dp.shape[0]} data points)")
                        
                        # Delete everything in the current recon_path folder
                        if os.path.exists(recon_path):
                            print(f"Deleting contents of {recon_path} before retrying...")
                            try:
                                # Walk through all files and directories in recon_path
                                for root, dirs, files in os.walk(recon_path, topdown=False):
                                    # First remove all files
                                    for file in files:
                                        file_path = os.path.join(root, file)
                                        os.unlink(file_path)
                                    # Then remove all directories
                                    for dir in dirs:
                                        dir_path = os.path.join(root, dir)
                                        os.rmdir(dir_path)
                                # Finally remove the main directory
                                os.rmdir(recon_path)
                                print(f"Successfully deleted {recon_path}")
                            except Exception as e:
                                print(f"Error while deleting {recon_path}: {str(e)}")

                        # Reinitialize the reconstruction path and task
                        recon_path = create_reconstruction_path(params, options)
                        save_initial_conditions(recon_path, params, options)
                        
                        # Clear CUDA cache before retrying
                        torch.cuda.empty_cache()
                        time.sleep(5)  # Longer delay to ensure memory is freed
                        
                        # Create a fresh task with the new options
                        task = PtychographyTask(options)
                    else:
                        # If it's not an out of memory error, re-raise the exception
                        print(f"Encountered non-memory error: {error_msg}")
                        raise
        else: # try recon once with fixed batch size
            for i in range(params['number_of_iterations'] // params['save_freq_iterations']):
                task.run(params['save_freq_iterations'])
                save_reconstructions(task, recon_path, params['save_freq_iterations']*(i+1), params)
                            
        return task, recon_path, params
    
    finally: # does seem to work
        # Ensure GPU memory is cleaned up even if an exception occurs
        torch.cuda.empty_cache()
        # Release all memory currently held
        #gc.collect()

        # Reset the CUDA device
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        # If you need a complete reset, you can also:
        torch.cuda.empty_cache()
        torch._C._cuda_clearCublasWorkspaces()

        # Allow time for complete cleanup
        #time.sleep(5)

def ptycho_batch_recon(base_params):
    """
    Process a range of ptychography scans with automatic error handling and status tracking.
    
    Args:
        base_params: Dictionary of parameters to use as a template for all scans
            start_scan: First scan number to process
            end_scan: Last scan number to process (inclusive)
            log_dir_suffix: Optional suffix for the log directory
            scan_order: Order to process the scans ('ascending', 'descending', or 'random')
            exclude_scans: List of scan numbers to exclude from processing
            overwrite_ongoing: Whether to overwrite scans marked as ongoing
            reset_scan_list: Whether to reset the scan list and process all scans again
            
    The function creates a tracker to monitor the status of each scan and processes
    them according to the specified order, skipping completed scans unless forced to reprocess.
    """
    # Extract parameters from base_params
    start_scan = base_params.get('start_scan')
    end_scan = base_params.get('end_scan')
    log_dir_suffix = base_params.get('log_dir_suffix', '')
    scan_order = base_params.get('scan_order', 'ascending')
    exclude_scans = base_params.get('exclude_scans', [])
    overwrite_ongoing = base_params.get('overwrite_ongoing', False)
    reset_scan_list = base_params.get('reset_scan_list', False)

    log_dir = os.path.join(base_params['data_directory'], 'ptychi_recons', 
                          base_params['recon_parent_dir'], 
                          f'recon_logs_{log_dir_suffix}' if log_dir_suffix else 'recon_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create tracker
    tracker = FileBasedTracker(log_dir, overwrite_ongoing=overwrite_ongoing)
    
    # Generate a unique worker ID
    worker_id = f"worker_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    
    num_repeats = np.inf
    repeat_count = 0
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
            
            # Try to start reconstruction
            if not tracker.start_recon(scan_num, worker_id, scan_params):
                print(f"Could not acquire lock for scan {scan_num}, skipping")
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
                    process = subprocess.Popen(
                        [sys.executable, script_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,  # Line buffered
                    )
                    
                    # Stream and print the output in real-time
                    for line in iter(process.stdout.readline, ''):
                        #print(line, end='')  # Already has newline
                        print(f"[S{scan_num:04d}-GPU{scan_params['gpu_id']}]{line}", end='')

                    # Wait for process to complete and get return code
                    return_code = process.wait()
                    
                    if return_code != 0:
                        raise subprocess.CalledProcessError(return_code, f"{sys.executable} {script_path}")
                    
                    # If we reached here, reconstruction was successful
                    elapsed_time = time.time() - start_time
                    print(f"Scan {scan_num} completed successfully in {elapsed_time:.2f} seconds")
                    successful_scans.append(scan_num)
                    
                    # Update status
                    tracker.complete_recon(scan_num, success=True)
                    
                except subprocess.CalledProcessError as e:
                    # Handle subprocess failure
                    elapsed_time = time.time() - start_time
                    error_message = f"Subprocess failed with exit code {e.returncode}"
                    
                    print(f"Scan {scan_num} failed after {elapsed_time:.2f} seconds with error: {error_message}")
                    failed_scans.append((scan_num, error_message))
                    
                    # Update status
                    tracker.complete_recon(scan_num, success=False, error=error_message)
                finally:
                    # Optionally remove the temporary files when done
                    # Uncomment these lines if you want to clean up after successful runs
                    if os.path.exists(params_path):
                        os.unlink(params_path)
                    if os.path.exists(script_path):
                        os.unlink(script_path)
                    
                    # Give system time to fully clean up resources
                    print(f"Waiting for 5 seconds before next scan...")
                    time.sleep(5)

                if reset_scan_list:
                    # Break the for loop after processing the current scan
                    break
                    
            except Exception as e:
                # Handle failure
                elapsed_time = time.time() - start_time
                print(f"Scan {scan_num} failed after {elapsed_time:.2f} seconds with error: {str(e)}")
                failed_scans.append((scan_num, str(e)))
                
                # Update status
                tracker.complete_recon(scan_num, success=False, error=str(e))
        
        # Print summary of processing
        #print(f"Successfully processed scans: {successful_scans}")
        print(f"Number of completed scans:{len(successful_scans)}/{end_scan - start_scan + 1}.")
        print(f"Number of failed scans:{len(failed_scans)}.")
        print(f"Number of ongoing scans:{len(ongoing_scans)}.")
        
        if len(successful_scans) == end_scan - start_scan + 1:
            print(f"All scans completed successfully")
            break
        else:
            repeat_count += 1
            print(f"Waiting for 5 seconds...")
            time.sleep(5)
   
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

def ptycho_batch_recon2(base_params):
    """
    Process multiple ptychography scans in parallel with automatic error handling and status tracking.
    
    Args:
        base_params: Dictionary of parameters to use as a template for all scans
            start_scan: First scan number to process
            end_scan: Last scan number to process (inclusive)
            log_dir_suffix: Optional suffix for the log directory
            scan_order: Order to process the scans ('ascending', 'descending', or 'random')
            exclude_scans: List of scan numbers to exclude from processing
            overwrite_ongoing: Whether to overwrite scans marked as ongoing
            max_workers: Maximum number of parallel processes (default: number of available GPUs)
            gpu_ids: List of GPU IDs to use (default: [0])
            print_interval: Interval to print the most recent line (default: 5 seconds)
            reset_scan_list: Whether to restart from the beginning of the scan list after each reconstruction (default: False)
    """
    from concurrent.futures import ThreadPoolExecutor
    import subprocess
    import sys
    import json
    import time
    import uuid
    import torch
    
    # Extract parameters with defaults
    start_scan = base_params.get('start_scan')
    end_scan = base_params.get('end_scan')
    log_dir_suffix = base_params.get('log_dir_suffix', '')
    scan_order = base_params.get('scan_order', 'ascending')
    exclude_scans = base_params.get('exclude_scans', [])
    overwrite_ongoing = base_params.get('overwrite_ongoing', False)
    gpu_ids = base_params.get('gpu_ids', [0])
    max_workers = base_params.get('max_workers', len(gpu_ids))
    print_interval = base_params.get('print_interval', 5)

    # Setup log directory
    log_dir = os.path.join(base_params['data_directory'], 'ptychi_recons', 
                          base_params['recon_parent_dir'], 
                          f'recon_logs_{log_dir_suffix}' if log_dir_suffix else 'recon_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create tracker
    tracker = FileBasedTracker(log_dir, overwrite_ongoing=overwrite_ongoing)
    
    # Setup temp directory
    temp_dir = os.path.join(base_params['data_directory'], 'ptychi_recons', 'temp_files')
    os.makedirs(temp_dir, exist_ok=True)
    
    def run_single_reconstruction(scan_num, gpu_id):
        """Run a single reconstruction on specified GPU"""
        worker_id = f"worker_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        
        # Create scan-specific parameters
        scan_params = base_params.copy()
        scan_params['scan_num'] = scan_num
        scan_params['gpu_id'] = gpu_id
        
        # Check status using tracker
        status = tracker.get_status(scan_num)
        if status == 'done':
            print(f"Scan {scan_num} already completed, skipping reconstruction")
            return 'success', scan_num, None
        if status == 'ongoing' and not overwrite_ongoing:
            print(f"Scan {scan_num} already ongoing, skipping reconstruction")
            return 'ongoing', scan_num, None
        
        # Try to start reconstruction
        if not tracker.start_recon(scan_num, worker_id, scan_params):
            print(f"Could not acquire lock for scan {scan_num}, skipping")
            return 'locked', scan_num, None
        
        print(f"\033[91mStarting reconstruction for scan {scan_num} on GPU {gpu_id}\033[0m")
        start_time = time.time()
        
        # Create temp file paths
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
            
            # Create reconstruction script
            script_content = f"""
import json
import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ptychi.pear import ptycho_recon

with open('{params_path}', 'r') as f:
    params = json.load(f)

# Convert lists back to NumPy arrays
for key, value in params.items():
    if isinstance(value, list) and key in ['scan_positions', 'positions']:
        params[key] = np.array(value)

ptycho_recon(run_recon=True, **params)
"""
            with open(script_path, 'w') as script_file:
                script_file.write(script_content)
            
            # Run reconstruction subprocess
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            last_print_time = time.time()
            output_buffer = []
            error_output = []
            
            # Stream output in real-time with GPU identifier
            while True:
                # Read from stdout
                stdout_line = process.stdout.readline()
                if stdout_line:
                    output_buffer.append(stdout_line)
                    current_time = time.time()
                    if current_time - last_print_time > print_interval:
                        if output_buffer:
                            print(f"[S{scan_num:04d}-GPU{gpu_id}]{output_buffer[-1]}", end='')
                        output_buffer = []
                        last_print_time = current_time
                
                # Read from stderr
                stderr_line = process.stderr.readline()
                if stderr_line:
                    error_output.append(stderr_line)
                
                # Check if process has finished
                if process.poll() is not None:
                    # Read any remaining output
                    remaining_stdout, remaining_stderr = process.communicate()
                    if remaining_stdout:
                        output_buffer.extend(remaining_stdout.splitlines())
                    if remaining_stderr:
                        error_output.extend(remaining_stderr.splitlines())
                    break
            
            return_code = process.returncode
            if return_code != 0:
                error_message = "".join(error_output) if error_output else "Process failed with no error output"
                raise subprocess.CalledProcessError(return_code, f"{sys.executable} {script_path}", error_message)
            
            elapsed_time = time.time() - start_time
            print(f"Scan {scan_num} completed successfully in {elapsed_time:.2f} seconds")
            tracker.complete_recon(scan_num, success=True)
            return 'success', scan_num, None
            
        except subprocess.CalledProcessError as e:
            elapsed_time = time.time() - start_time
            error_message = e.stderr if hasattr(e, 'stderr') else str(e)
            print(f"Scan {scan_num} failed after {elapsed_time:.2f} seconds with error: {error_message}")
            tracker.complete_recon(scan_num, success=False, error=error_message)
            return 'failed', scan_num, error_message
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_message = str(e)
            print(f"Scan {scan_num} failed after {elapsed_time:.2f} seconds with error: {error_message}")
            tracker.complete_recon(scan_num, success=False, error=error_message)
            return 'failed', scan_num, error_message
            
        finally:
            # Clean up temporary files
            for path in [params_path, script_path]:
                if os.path.exists(path):
                    os.unlink(path)
            
            # Ensure GPU memory is cleaned up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Generate scan list based on order
    scan_list = generate_scan_list(start_scan, end_scan, scan_order, exclude_scans)
    
    # Process scans in parallel
    successful_scans = []
    failed_scans = []
    ongoing_scans = []
    
    print(f"Processing {len(scan_list)} scans using {max_workers} workers on GPUs: {gpu_ids}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks for each scan, cycling through available GPUs
        futures = []
        gpu_usage = {}  # Track which GPUs are in use
        
        for scan_num in scan_list:
            # Check status using tracker
            status = tracker.get_status(scan_num)
            if status == 'done':
                print(f"Scan {scan_num} already completed, skipping reconstruction")
                successful_scans.append(scan_num)
                continue
            if status == 'ongoing' and not overwrite_ongoing:
                print(f"Scan {scan_num} already ongoing, skipping reconstruction")
                ongoing_scans.append(scan_num)
                continue
            
            # Wait for an available GPU
            available_gpus = wait_for_available_gpu(gpu_ids, gpu_usage)
            
            # Select the best available GPU
            gpu_id = select_gpu(gpu_list=available_gpus)
            gpu_usage[gpu_id] = scan_num  # Mark GPU as in use
            
            print(f"Starting scan {scan_num} on GPU {gpu_id}")
            future = executor.submit(run_single_reconstruction, scan_num, gpu_id)
            futures.append((future, gpu_id))
            time.sleep(1)  # Small delay between submissions
        
        # Collect results and update GPU usage
        for future, gpu_id in futures:
            try:
                status, scan_num, error = future.result()
                if status == 'success':
                    successful_scans.append(scan_num)
                elif status == 'failed':
                    failed_scans.append((scan_num, error))
                elif status == 'ongoing':
                    ongoing_scans.append(scan_num)
                
                if status in ['success', 'failed']:
                    del gpu_usage[gpu_id]  # Free up the GPU
            except Exception as e:
                print(f"Error processing scan on GPU {gpu_id}: {str(e)}")
                if gpu_id in gpu_usage:
                    del gpu_usage[gpu_id]  # Free up the GPU even if there was an error
    
    # Print summary
    print_summary(successful_scans, failed_scans, ongoing_scans, start_scan, end_scan)
    
    return successful_scans, failed_scans, ongoing_scans

def wait_for_available_gpu(gpu_ids, gpu_usage, wait_time=5):
    """Wait until at least one GPU is available and return the list of available GPUs."""
    available_gpus = [gpu_id for gpu_id in gpu_ids if gpu_id not in gpu_usage]
    
    if not available_gpus:
        print("Waiting for a GPU to become available...")
        while not available_gpus:
            time.sleep(wait_time)
            available_gpus = [gpu_id for gpu_id in gpu_ids if gpu_id not in gpu_usage]
    
    return available_gpus

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
