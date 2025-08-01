import random
import os
import datetime
import logging

import torch
import h5py
import numpy as np
import pytest

from ptychi.global_settings import set_default_complex_dtype
from ptychi.utils import rescale_probe, add_additional_opr_probe_modes_to_probe, to_tensor
import ptychi.timing.timer_utils as timer_utils
from ptychi.timing.io import save_timing_data_to_unique_file_path


class BaseTester:
    def setup_method(
        self,
        name="",
        generate_data=False,
        generate_gold=False,
        debug=False,
        action=None,
        pytestconfig=None,
    ):
        """
        A Pytest hook that sets instance attributes before running each test method. 
        If the script is executed with `python`, this method will not run automatically
        before calling a test method. Therefore, it can be used to set instance attributes
        for all methods in a code snippet if that snippet is intended to be executed
        with `python`.

        Parameters
        ----------
        name : str, optional
            The name of the tester.
        generate_data : bool
            Whether to generate test data. 
        generate_gold : bool
            Whether to generate gold data. 
        save_timing : bool
            Whether to save timing results.
        debug : bool, optional
            Switches debug mode.
        """
        logging.basicConfig(level=logging.INFO)
        
        self.name = name
        
        self.generate_data = generate_data
        self.generate_gold = generate_gold
        # self.save_timing = save_timing
        self.debug = debug
        
        if pytestconfig is not None:
            self.high_tol = pytestconfig.getoption("high_tol")
            self.action = pytestconfig.getoption("action")
            self.save_timing = pytestconfig.getoption("save_timing")
        else:
            self.high_tol = False
            self.action = action
            self.save_timing = False
            
        self.atol = 1e-3
        self.rtol = 1e-2 if self.high_tol else 1e-4
    
    @pytest.fixture(autouse=True)
    def inject_config(self, pytestconfig):
        self.pytestconfig = pytestconfig
        self.setup_method(
            name="",
            generate_data=False,
            generate_gold=False,
            debug=False,
            action=None,
            pytestconfig=pytestconfig,
        )
    
    @staticmethod
    def get_ci_data_dir():
        try:
            dir = os.environ['PTYCHO_CI_DATA_DIR']
        except KeyError:
            raise KeyError('PTYCHO_CI_DATA_DIR not set. Please set it to the path to the data folder.')
        return dir

    def get_ci_input_data_dir(self):
        return os.path.join(self.get_ci_data_dir(), 'data')

    def get_ci_gold_data_dir(self):
        return os.path.join(self.get_ci_data_dir(), 'gold_data')
    
    """
    @property
    def gold_run_id(self) -> int:
        return int(self.ci_config['TESTING']['GoldRunID'])
        
    def get_run_id(self) -> int:
        \"""
        Get the current run ID as the maximum of the existing run IDs in the output directory plus 1.
        If the output directory does not contain any run IDs, return 1.

        Returns
        -------
        int
            The current run ID.
        \"""
        output_dir = self.get_output_dir()
        run_ids = [int(f.name) for f in os.scandir(output_dir) if f.is_dir()]
        if len(run_ids) == 0:
            return 1
        return max(run_ids) + 1
        
    @staticmethod
    def get_input_dir():
        try:
            dir = os.environ['PTYCHI_CI_INPUT_DIR']
        except KeyError:
            raise KeyError('PTYCHI_CI_INPUT_DIR not set. Please set it to the path to the data folder.')
        return dir
    
    @staticmethod
    def get_output_dir():
        try:
            dir = os.environ['PTYCHI_CI_OUTPUT_DIR']
        except KeyError:
            raise KeyError('PTYCHI_CI_OUTPUT_DIR not set. Please set it to the path to the data folder.')
        return dir
    
    def get_ci_config(self):
        try:
            config_ini = os.environ['PTYCHI_CI_CONFIG_PATH']
        except KeyError:
            raise KeyError('PTYCHI_CI_CONFIG_PATH not set. Please set it to the path to the config.ini file.')
        config = configparser.ConfigParser()
        config.read(config_ini)
        return config
    """
    
    def save_gold_data(self, name, data):
        fname = os.path.join(self.get_ci_gold_data_dir(), name, 'recon.npy')
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        np.save(fname, data)
    
    def load_gold_data(self, name):
        fname = os.path.join(self.get_ci_gold_data_dir(), name, 'recon.npy')
        return np.load(fname)
    
    def run_comparison(self, name, test_data, atol=None, rtol=None):
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        gold_data = self.load_gold_data(name)
        compare_data(test_data, gold_data, atol=atol, rtol=rtol, name=name)
        return
    
    def get_default_input_data_file_paths(self, name):
        dp = os.path.join(self.get_ci_input_data_dir(), name, 'ptychodus_dp.hdf5')
        para = os.path.join(self.get_ci_input_data_dir(), name, 'ptychodus_para.hdf5')
        return dp, para

    def setup_ptychi(self, cpu_only=False, gpu_indices=(0,)):
        torch.manual_seed(123)
        random.seed(123)
        np.random.seed(123)
        
        torch.set_default_device('cpu' if cpu_only else 'cuda')
        torch.set_default_dtype(torch.float32)
        set_default_complex_dtype(torch.complex64)
        
        if not cpu_only:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_indices))

    def load_data_ptychodus(self, diffraction_pattern_file, parameter_file, subtract_position_mean=False, additional_opr_modes=0, scale_probe_magnitude=True):
        patterns = h5py.File(diffraction_pattern_file, 'r')['dp'][...]

        f_meta = h5py.File(parameter_file, 'r')
        probe = f_meta['probe'][...]
        if scale_probe_magnitude:
            probe = rescale_probe(probe, patterns)
        if probe.ndim == 3:
            probe = probe[None, :, :, :]
        probe = to_tensor(probe)
        if additional_opr_modes > 0:
            probe = add_additional_opr_probe_modes_to_probe(probe, n_opr_modes_to_add=additional_opr_modes)
        
        positions = np.stack([f_meta['probe_position_y_m'][...], f_meta['probe_position_x_m'][...]], axis=1)
        pixel_size_m = f_meta['object'].attrs['pixel_height_m']
        positions_px = positions / pixel_size_m
        if subtract_position_mean:
            positions_px -= positions_px.mean(axis=0)
        
        return patterns, probe, pixel_size_m, positions_px
    
    def plot_object(self, obj):
        if obj.ndim == 2:
            obj = obj[None, ...]
        if obj.shape[0] > 1:
            plot_multislice_phase(obj)
        else:
            plot_complex_image(obj)
    
    @staticmethod
    def wrap_recon_tester(name="", run_comparison=True):
        """
        A decorator factory that wraps a test method to generate or compare data.

        Parameters
        ----------
        name : str, optional
            The name of the test.
        """
        def decorator(test_method):
            def wrapper(self: BaseTester):
                if self.save_timing:
                    timer_utils.toggle_timer(enable=True)
                recon = test_method(self)
                if self.debug and not self.generate_gold:
                    self.plot_object(recon)
                if self.generate_gold:
                    self.save_gold_data(name, recon)
                if not self.generate_gold and run_comparison:
                    self.run_comparison(name, recon, atol=self.atol, rtol=self.rtol)
                if self.save_timing:
                    save_timing_data_to_unique_file_path(name, get_timing_data_dir())
            return wrapper
        return decorator
    

class TungstenDataTester(BaseTester):
    
    def load_tungsten_data(self, additional_opr_modes=0, pos_type='true'):
        return self.load_data_ptychodus(
            os.path.join(self.get_ci_input_data_dir(), '2d_ptycho', 'dp_250.hdf5'), 
            os.path.join(self.get_ci_input_data_dir(), '2d_ptycho', 'metadata_250_{}Pos.hdf5'.format(pos_type)), 
            additional_opr_modes=additional_opr_modes
        )


def get_timestamp():
    return datetime.datetime.now().strftime('%Y%m%d%H%M')


def compare_data(test_data, gold_data, atol=1e-7, rtol=1e-7, name=""):
    if not np.allclose(gold_data.shape, test_data.shape):
        print('{} FAILED [SHAPE MISMATCH]'.format(name))
        print('  Gold shape: {}'.format(gold_data.shape))
        print('  Test shape: {}'.format(test_data.shape))
        raise AssertionError
    if not np.allclose(gold_data, test_data, atol=atol, rtol=rtol):
        print('{} FAILED [MISMATCH]'.format(name))
        abs_diff = np.abs(gold_data - test_data)
        loc_max_diff = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        loc_max_diff = [a.item() for a in loc_max_diff]
        print('  Mean abs diff: {}'.format(abs_diff.mean()))
        print('  Location of max diff: {}'.format(loc_max_diff))
        print('  Max abs diff: {}'.format(abs_diff[*loc_max_diff].item()))
        print('  Value at max abs diff (test): {}'.format(test_data[*loc_max_diff].item()))
        print('  Value at max abs diff (gold): {}'.format(gold_data[*loc_max_diff].item()))
        raise AssertionError
    print('{} PASSED'.format(name))


def plot_complex_image(img):
    import matplotlib.pyplot as plt
    if img.ndim == 3:
        img = img[0]
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(np.abs(img))
    ax[0].set_title('magnitude')
    ax[1].imshow(np.angle(img))
    ax[1].set_title('phase')
    plt.show()


def plot_multislice_phase(img):
    import matplotlib.pyplot as plt
    n_slices = img.shape[0]
    fig, ax = plt.subplots(n_slices, 1)
    for i in range(n_slices):
        ax[i].imshow(np.angle(img[i, ...]))
        ax[i].set_title('slice {}'.format(i))
    plt.show()
    
    
def plot_probe_positions(pos, pos_true=None):
    import matplotlib.pyplot as plt
    plt.scatter(pos[:, 1], pos[:, 0], label="reconstructed")
    if pos_true is not None:
        plt.scatter(pos_true[:, 1], pos_true[:, 0], label="true")
    plt.legend()
    plt.show()


def get_timing_data_dir():
    return os.path.join(BaseTester.get_ci_data_dir(), "timing")