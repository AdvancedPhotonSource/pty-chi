import subprocess
import re
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import json
import tempfile
import shutil
import fcntl
from datetime import datetime

def select_gpu(gpu_list=None):
    # Get GPU stats
    gpu_stats = get_gpu_usage()

    if gpu_stats is None or len(gpu_stats) == 0:
        # Default to GPU 0 if can't get stats
        return 0
    else:
        # Filter GPUs based on gpu_list if provided
        if gpu_list is not None:
            gpu_stats = [gpu for gpu in gpu_stats if gpu['index'] in gpu_list]
            if not gpu_stats:
                print(f"No GPUs from the provided list are available. Defaulting to GPU 0.")
                return 0
        
        # Find GPU with 0% utilization and lowest memory usage
        zero_util_gpus = [gpu for gpu in gpu_stats if gpu['utilization'] == 0]
        
        if zero_util_gpus:
            # Among GPUs with 0% utilization, pick the one with lowest memory usage
            selected_gpu = min(zero_util_gpus, key=lambda x: x['memory_percent'])
        else:
            print(f"No GPU has 0% utilization. Selecting GPU with lowest combined score.")
            # If no GPU has 0% utilization, pick the one with lowest combined score
            selected_gpu = min(gpu_stats, 
                                key=lambda x: x['utilization'] + x['memory_percent'])
        
        return selected_gpu['index']

# Check GPU usage and select least utilized GPU
def get_gpu_usage():
    try:
        # Run nvidia-smi to get GPU utilization and memory usage
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'])
        output = output.decode('utf-8').strip().split('\n')
        
        gpu_stats = []
        for line in output:
            index, util, mem_used, mem_total = line.split(', ')
            gpu_stats.append({
                'index': int(index),
                'utilization': int(util),
                'memory_used': int(mem_used),
                'memory_total': int(mem_total),
                'memory_percent': (int(mem_used) / int(mem_total)) * 100
            })
        return gpu_stats
    except:
        return None

def check_gpu_availability(gpu_ids, min_free_memory_mb=1000):
    """
    Check which GPUs from the list are actually available with sufficient memory.
    
    Args:
        gpu_ids: List of GPU IDs to check
        min_free_memory_mb: Minimum free memory required in MB
        
    Returns:
        List of available GPU IDs
    """
    try:
        gpu_stats = get_gpu_usage()
        if not gpu_stats:
            return gpu_ids  # Fallback if can't get stats
        
        available_gpus = []
        for gpu in gpu_stats:
            if gpu['index'] in gpu_ids:
                free_memory_mb = gpu['memory_total'] - gpu['memory_used']
                if gpu['utilization'] < 50 and free_memory_mb > min_free_memory_mb:
                    available_gpus.append(gpu['index'])
        
        return available_gpus if available_gpus else gpu_ids  # Fallback to original list
    except Exception as e:
        print(f"Warning: Could not check GPU availability: {e}")
        return gpu_ids  # Fallback to original list

def make_fzp_probe(N, lambda_, dx, Ls, Rn, dRn, D_FZP, D_H):

    """
    Generates a Fresnel zone plate probe.
    Parameters:
        N (int): Number of pixels.
        lambda_ (float): Wavelength.
        dx (float): Pixel size (in meters) in sample plane.
        Ls (float): Distance (in meters) from focal plane to sample.
        Rn (float): Radius of outermost zone (in meters).
        dRn (float): Width of outermost zone (in meters).
        D_FZP (float): Diameter of pinhole.
        D_H (float): Diameter of the central beamstop (in meters).
    Returns:
        ndarray: Calculated probe field in the sample plane.
    """

    fl = 2 * Rn * dRn / lambda_  # focal length corresponding to central wavelength
    dx_fzp = lambda_ * fl / N / dx  # pixel size in the FZP plane
    # Coordinate in the FZP plane
    lx_fzp = np.linspace(-dx_fzp * N / 2, dx_fzp * N / 2, N)
    x_fzp, y_fzp = np.meshgrid(lx_fzp, lx_fzp)
    # Transmission function of the FZP
    T = np.exp(-1j * 2 * np.pi / lambda_ * (x_fzp**2 + y_fzp**2) / (2 * fl))
    C = (np.sqrt(x_fzp**2 + y_fzp**2) <= (D_FZP / 2)).astype(np.float64)  # circular function of FZP
    H = (np.sqrt(x_fzp**2 + y_fzp**2) >= (D_H / 2)).astype(np.float64)  # central block
    # Probe on sample plane using the Fresnel propagation function defined previously
    probe = fresnel_propagation(C * T * H, dx_fzp, fl+Ls, lambda_)
    #print(f"dx_fzp: {dx_fzp}, fl: {fl}, Ls: {Ls}, lambda_: {lambda_}")
    #print(f"Center pixel value of probe: {probe[probe.shape[0]//2, probe.shape[1]//2]}")

    return probe

def fresnel_propagation(IN, dxy, z, lambda_):
    """
    Performs Fresnel propagation for a given input wavefield.
    
    Parameters:
        IN (ndarray): Input object or wavefield.
        dxy (float): Pixel pitch of the object.
        z (float): Distance of the propagation.
        lambda_ (float): Wavelength of the wave.
    
    Returns:
        ndarray: Output wavefield after Fresnel propagation.
    """
    M, N = IN.shape
    k = 2 * np.pi / lambda_

    # Coordinate grid for input plane
    lx = np.linspace(-dxy * M / 2, dxy * M / 2, M)
    x, y = np.meshgrid(lx, lx)

    # Coordinate grid for output plane
    fc = 1 / dxy
    fu = lambda_ * z * fc
    lu = np.fft.ifftshift(np.linspace(-fu / 2, fu / 2, M))
    u, v = np.meshgrid(lu, lu)

    if z > 0:
        # Propagation in the positive z direction
        pf = np.exp(1j * k * z) * np.exp(1j * k * (u**2 + v**2) / (2 * z))
        kern = IN * np.exp(1j * k * (x**2 + y**2) / (2 * z))
        
        kerntemp = np.fft.fftshift(kern)
        cgh = np.fft.fft2(kerntemp)
        OUT = np.fft.fftshift(cgh * pf)
    else:
        # Propagation in the negative z direction (or backward propagation)
        z = abs(z)
        pf = np.exp(1j * k * z) * np.exp(1j * k * (x**2 + y**2) / (2 * z))
        cgh = np.fft.ifft2(np.fft.ifftshift(IN) / np.exp(1j * k * (u**2 + v**2) / (2 * z)))
        OUT = np.fft.fftshift(cgh) / pf
    
    return OUT

def resize_complex_array(complex_array, new_shape=None, zoom_factor=None):
    """
    Resize a complex array using scipy.ndimage.zoom.
    
    Parameters:
    -----------
    complex_array : ndarray
        The complex array to resize
    new_shape : tuple, optional
        The target shape (height, width). Required if zoom_factor is None.
    zoom_factor : float or tuple, optional
        The zoom factor to apply. Can be a single float or a tuple for different factors per dimension.
        Required if new_shape is None.
    
    Returns:
    --------
    ndarray
        The resized complex array
    """
    import numpy as np
    import scipy.ndimage
    
    # Input validation
    if new_shape is None and zoom_factor is None:
        raise ValueError("Either new_shape or zoom_factor must be provided")
    
    # Handle different array dimensions
    if complex_array.ndim == 2:
        # Calculate zoom_factor from new_shape if needed
        if zoom_factor is None:
            zoom_factor = (new_shape[0] / complex_array.shape[0], 
                          new_shape[1] / complex_array.shape[1])
        
        # Resize 2D array
        real_part = scipy.ndimage.zoom(complex_array.real, zoom_factor, order=1)
        imag_part = scipy.ndimage.zoom(complex_array.imag, zoom_factor, order=1)
        return real_part + 1j * imag_part
    
    elif complex_array.ndim == 3:
        # For 3D arrays, resize each 2D slice separately
        if new_shape is None:
            if isinstance(zoom_factor, (int, float)):
                new_shape = (int(complex_array.shape[1] * zoom_factor),
                            int(complex_array.shape[2] * zoom_factor))
            else:  # zoom_factor is a tuple
                if len(zoom_factor) == 2:
                    new_shape = (int(complex_array.shape[1] * zoom_factor[0]),
                                int(complex_array.shape[2] * zoom_factor[1]))
                else:
                    new_shape = (int(complex_array.shape[1] * zoom_factor[1]),
                                int(complex_array.shape[2] * zoom_factor[2]))
        
        # Create output array
        result = np.zeros((complex_array.shape[0], new_shape[0], new_shape[1]), dtype=complex)
        
        # Calculate 2D zoom factor for each slice
        slice_zoom_factor = (new_shape[0] / complex_array.shape[1], 
                            new_shape[1] / complex_array.shape[2])
        
        # Process each slice
        for i in range(complex_array.shape[0]):
            real_part = scipy.ndimage.zoom(complex_array[i].real, slice_zoom_factor, order=1)
            imag_part = scipy.ndimage.zoom(complex_array[i].imag, slice_zoom_factor, order=1)
            result[i] = real_part + 1j * imag_part
        
        return result
    
    else:
        raise ValueError(f"Unsupported array dimension: {complex_array.ndim}")

# potential alternative to resize_complex_array
def resize_complex_fourier(complex_array, new_shape):
    # Get dimensions
    old_shape = complex_array.shape
    
    # Compute FFT of the input array
    fft_array = np.fft.fftshift(np.fft.fft2(complex_array))
    
    # Create a new array for the resized FFT
    new_fft = np.zeros(new_shape, dtype=complex)
    
    # Calculate padding or cropping dimensions
    pad_x = (new_shape[1] - old_shape[1]) // 2
    pad_y = (new_shape[0] - old_shape[0]) // 2
    
    # Copy the central portion of the FFT
    if pad_x >= 0 and pad_y >= 0:  # Upsampling
        new_fft[pad_y:pad_y+old_shape[0], pad_x:pad_x+old_shape[1]] = fft_array
    else:  # Downsampling
        new_fft = fft_array[-pad_y:old_shape[0]+pad_y, -pad_x:old_shape[1]+pad_x]
    
    # Compute inverse FFT
    return np.fft.ifft2(np.fft.ifftshift(new_fft))

def near_field_evolution(u_0, z, wavelength, extent, use_ASM_only=False):
    """
    Near-field evolution function for wave propagation.
    Automatically switches between ASM and Fraunhofer propagation.
    
    Parameters:
    -----------
    u_0 : ndarray
        Input wavefield
    z : float
        Propagation distance
    wavelength : float
        Wavelength of the wave
    extent : float or tuple
        Physical size of the array (in meters)
    use_ASM_only : bool, optional
        Force using Angular Spectrum Method only
        
    Returns:
    --------
    u_1 : ndarray
        Propagated wavefield
    H : ndarray
        Transfer function
    h : ndarray
        Impulse response
    dH : ndarray
        Derivative of transfer function
    """
    import numpy as np
    
    # Initialize return values
    H = None
    h = None
    u_1 = None
    dH = None
    
    # Handle extent as a scalar or tuple
    if np.isscalar(extent):
        extent = np.array([extent, extent])
    else:
        extent = np.array(extent)
    
    # If z is 0, return input field unchanged
    if z == 0:
        H = 1
        u_1 = u_0
        return u_1, H, h, dH
    
    # If z is infinity, return empty arrays
    if np.isinf(z):
        return u_1, H, h, dH
    
    # Get dimensions of input field
    Npix = u_0.shape
    
    # Create normalized coordinate grids
    xgrid = (0.5 + np.arange(-Npix[0]//2, Npix[0]//2)) / Npix[0]
    ygrid = (0.5 + np.arange(-Npix[1]//2, Npix[1]//2)) / Npix[1]
    
    # Wave number
    k = 2 * np.pi / wavelength
    
    # Undersampling parameter
    F = np.mean(extent**2 / (wavelength * z * np.array(Npix)))
    
    if abs(F) < 1 and not use_ASM_only:
        # Farfield propagation
        print(f'Farfield regime, F/Npix={F}')
        Xrange = xgrid * extent[0]
        Yrange = ygrid * extent[1]
        X, Y = np.meshgrid(Xrange, Yrange)
        h = np.exp(1j * k * z + 1j * k / (2 * z) * (X.T**2 + Y.T**2))
        
        # This serves as low pass filter for the far nearfield
        H = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(h)))
        # Renormalize to conserve flux in image
        H = H / np.abs(H[Npix[0]//2, Npix[1]//2])
    else:
        # Standard Angular Spectrum Method (ASM)
        kx = 2 * np.pi * xgrid / extent[0] * Npix[0]
        ky = 2 * np.pi * ygrid / extent[1] * Npix[1]
        Kx, Ky = np.meshgrid(kx, ky)
        
        dH = (-1j * (Kx.T**2 + Ky.T**2) / (2 * k))
        
        # Make it a bit more sensitive to z distance
        H = np.exp(1j * z * np.sqrt(k**2 - Kx.T**2 - Ky.T**2))
        h = None
    
    # Apply transfer function to input field
    u_1 = np.fft.ifft2(np.fft.ifftshift(H) * np.fft.fft2(u_0))
    
    return u_1, H, h, dH

def format_path_with_scan_num(path, scan_num):
    """
    Format a path string with scan number, handling various format specifiers.
    
    Parameters:
    -----------
    path : str
        Path string that may contain format specifiers like {scan_num} or {scan_num:04d}
    scan_num : int
        Scan number to insert into the path
        
    Returns:
    --------
    str
        Formatted path with scan number inserted
    """
    if not path:
        return path
        
    if '{scan_num' in path:  # if path contains any scan_num format
        try:
            formatted_path = path.format(scan_num=scan_num)
            return formatted_path
        except Exception as e:
            print(f"Warning: Failed to format path with scan_num: {str(e)}")
            return path
    else:
        return path

def find_matching_recon(path, scan_num):
    """
    Find a matching reconstruction file based on a path pattern and scan number.
    
    Parameters:
    -----------
    path : str
        Path pattern that may contain format specifiers
    scan_num : int
        Scan number to use for formatting
        
    Returns:
    --------
    str
        Path to the matching reconstruction file
    """
    formatted_path = format_path_with_scan_num(path, scan_num)
    
    matching_files = glob.glob(formatted_path)
    if matching_files:
        # Sort files by modification time (newest first)
        matching_files.sort(key=os.path.getmtime, reverse=True)
        return matching_files[0]
    else:
        raise FileNotFoundError(f"No matching reconstruction file found for pattern: {formatted_path}")

def generate_scan_list(start_scan, end_scan, scan_order='ascending', exclude_scans=None):
    """Generate a list of scan numbers based on the specified order and exclusions."""
    """Used for batch reconstruction"""
    
    if exclude_scans is None:
        exclude_scans = []
        
    if scan_order == 'ascending':
        scan_list = list(range(start_scan, end_scan + 1))
    elif scan_order == 'descending':
        scan_list = list(range(end_scan, start_scan - 1, -1))
    elif scan_order == 'random':
        import random
        scan_list = list(range(start_scan, end_scan + 1))
        random.shuffle(scan_list)
    else:
        raise ValueError(f"Invalid scan_order: {scan_order}. Must be 'ascending', 'descending', or 'random'.")
    
    return [scan for scan in scan_list if scan not in exclude_scans]

class FileBasedTracker:
    def __init__(self, base_dir, overwrite_ongoing=False):
        """Initialize the tracker with a base directory for status files."""
        self.base_dir = base_dir
        self.status_dir = os.path.join(base_dir, 'status')
        self.lock_dir = os.path.join(base_dir, 'locks')
        self.overwrite_ongoing = overwrite_ongoing
        
        # Create directories if they don't exist
        os.makedirs(self.status_dir, exist_ok=True)
        os.makedirs(self.lock_dir, exist_ok=True)
    
    def _get_status_file(self, scan_id):
        """Get the path to the status file for a scan."""
        return os.path.join(self.status_dir, f"scan_{scan_id:04d}.json")
    
    def _get_lock_file(self, scan_id):
        """Get the path to the lock file for a scan."""
        return os.path.join(self.lock_dir, f"scan_{scan_id:04d}.lock")
    
    def get_status(self, scan_id):
        """Get the current status of a scan."""
        status_file = self._get_status_file(scan_id)
        
        if not os.path.exists(status_file):
            return None
        
        try:
            with open(status_file, 'r') as f:
                data = json.load(f)
                return data.get('status')
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def get_full_status(self, scan_id):
        """Get the full status data of a scan including batch info and error type."""
        status_file = self._get_status_file(scan_id)

        if not os.path.exists(status_file):
            return None
        
        try:
            with open(status_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def start_recon(self, scan_id, worker_id, params):
        """
        Try to start a reconstruction for a scan.
        Returns True if successful, False if already in progress or completed.
        """
        lock_file = self._get_lock_file(scan_id)
        status_file = self._get_status_file(scan_id)
        
        # Create or open the lock file
        try:
            lock_fd = open(lock_file, 'w')
            # Try to acquire an exclusive, non-blocking lock
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, BlockingIOError):
            # Another process has the lock
            return False
        
        try:
            # Check if status file exists and scan is already done or in progress
            if os.path.exists(status_file):
                try:
                    with open(status_file, 'r') as f:
                        data = json.load(f)
                        if data.get('status') =='done':
                            return False
                        if data.get('status') == 'ongoing' and not self.overwrite_ongoing:
                            return False
                except (json.JSONDecodeError, FileNotFoundError):
                    # Corrupted or missing status file, we can proceed
                    pass
            
            # Create a temporary file first to avoid partial writes
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', 
                dir=self.status_dir,
                delete=False
            )
            
            # Prepare status data
            import socket
            import getpass
            status_data = {
                'status': 'ongoing',
                'host_name': socket.gethostname(),
                'user_name': getpass.getuser(),
                'scan_id': scan_id,
                'worker_id': worker_id,
                'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                
            }
            
            # Write to temporary file
            json.dump(status_data, temp_file)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()
            
            # Atomically move the temporary file to the final location
            shutil.move(temp_file.name, status_file)
            
            return True
            
        finally:
            # Release the lock
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
    
    def complete_recon(self, scan_id, success=True, error=None, number_of_batches=None):
        """Mark a reconstruction as completed or failed, optionally storing batch information."""
        lock_file = self._get_lock_file(scan_id)
        status_file = self._get_status_file(scan_id)
        
        # Acquire lock
        with open(lock_file, 'w') as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            
            try:
                # Read current status
                if os.path.exists(status_file):
                    with open(status_file, 'r') as f:
                        status_data = json.load(f)
                else:
                    # Create new status data if file doesn't exist
                    status_data = {
                        'scan_id': scan_id,
                        'start_time': 'unknown'
                    }
                
                # Update status
                status_data['status'] = 'done' if success else 'failed'
                status_data['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if error:
                    status_data['error'] = str(error)
                    # Classify error type
                    error_str = str(error).lower()
                    if any(keyword in error_str for keyword in 
                          ['cuda out of memory', 'outofmemory', 'cudaerroroutofmemory']):
                        status_data['error_type'] = 'out_of_memory'
                    elif any(keyword in error_str for keyword in 
                            ['cuda error', 'cudnn', 'cublas', 'device']):
                        status_data['error_type'] = 'gpu_error'
                    elif any(keyword in error_str for keyword in 
                            ['filenotfounderror', 'file not found', 'no such file']):
                        status_data['error_type'] = 'file_not_found'
                    elif 'timeout' in error_str:
                        status_data['error_type'] = 'timeout'
                    else:
                        status_data['error_type'] = 'unknown'
                
                # Store batch information if provided
                if number_of_batches is not None:
                    status_data['number_of_batches'] = number_of_batches
                
                # Write to temporary file first
                temp_file = tempfile.NamedTemporaryFile(
                    mode='w', 
                    dir=self.status_dir,
                    delete=False
                )
                
                # Write status data using json.dump for proper formatting and escaping
                json.dump(status_data, temp_file, indent=4)
                temp_file.flush()
                os.fsync(temp_file.fileno())
                temp_file.close()
                
                # Atomically move the temporary file to the final location
                shutil.move(temp_file.name, status_file)
                
            except Exception as e:
                print(f"Error updating status for scan {scan_id}: {str(e)}")

def crop_pad(img, outsize, fill=0):
    """
    Adjust the size by padding or cropping, centered.

    For 2D arrays, operates on (rows, cols).
    For 3D arrays, operates on the last two dimensions (leave the first dimension unchanged).
    For N>3, operates on the first two dimensions (backward compatible).

    Parameters
    ----------
    img : ndarray
        Input array (real or complex)
    outsize : tuple[int, int]
        Target size (height, width) for the spatial dimensions
    fill : scalar, optional
        Fill value for padding (default 0)

    Returns
    -------
    ndarray
        Cropped/padded array with spatial dimensions equal to outsize
    """
    if outsize is None:
        print('CROP_PAD: adjusts the size by zero padding or cropping')
        print('crop_pad(img, outsize)')
        return

    Nin = img.shape
    if len(outsize) < 2:
        raise ValueError('outsize must be a tuple/list of two integers (height, width)')

    # Determine which axes are spatial
    if img.ndim >= 2:
        # Default spatial axes are the last two
        spatial_axis_row = img.ndim - 2
        spatial_axis_col = img.ndim - 1
        # For N>3, preserve previous behavior (operate on first two dims)
        if img.ndim > 3:
            spatial_axis_row = 0
            spatial_axis_col = 1
    else:
        # Nothing to do for 0D/1D arrays
        return img

    in_rows = Nin[spatial_axis_row]
    in_cols = Nin[spatial_axis_col]

    # Early exit if already matching target spatial size
    if outsize[0] == in_rows and outsize[1] == in_cols:
        return img

    Nout = (int(outsize[0]), int(outsize[1]))

    # Build output shape by replacing spatial dims
    output_shape = list(Nin)
    output_shape[spatial_axis_row] = Nout[0]
    output_shape[spatial_axis_col] = Nout[1]

    imout = np.zeros(output_shape, dtype=img.dtype)
    if fill != 0:
        imout += fill

    # Centers (1-based intent in MATLAB; integer math below matches that behavior)
    center_in_row = in_rows // 2
    center_in_col = in_cols // 2
    center_out_row = Nout[0] // 2
    center_out_col = Nout[1] // 2

    delta_row = center_out_row - center_in_row
    delta_col = center_out_col - center_in_col

    # Compute source ranges
    src_row_start = max(-delta_row, 0)
    src_row_end   = min(-delta_row + Nout[0], in_rows)
    src_col_start = max(-delta_col, 0)
    src_col_end   = min(-delta_col + Nout[1], in_cols)

    # Compute destination ranges
    dst_row_start = max(delta_row, 0)
    dst_row_end   = min(delta_row + in_rows, Nout[0])
    dst_col_start = max(delta_col, 0)
    dst_col_end   = min(delta_col + in_cols, Nout[1])

    # Only proceed if there is an overlapping region
    if (src_row_end > src_row_start and src_col_end > src_col_start and
        dst_row_end > dst_row_start and dst_col_end > dst_col_start):
        src_slices = [slice(None)] * img.ndim
        dst_slices = [slice(None)] * img.ndim
        src_slices[spatial_axis_row] = slice(src_row_start, src_row_end)
        src_slices[spatial_axis_col] = slice(src_col_start, src_col_end)
        dst_slices[spatial_axis_row] = slice(dst_row_start, dst_row_end)
        dst_slices[spatial_axis_col] = slice(dst_col_start, dst_col_end)
        imout[tuple(dst_slices)] = img[tuple(src_slices)]

    # Preserve complex dtype if input is complex
    if np.iscomplexobj(img) and not np.iscomplexobj(imout):
        imout = imout.astype(complex)

    return imout
