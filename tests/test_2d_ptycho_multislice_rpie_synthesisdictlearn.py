import argparse
import os

import torch
import numpy as np

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_suggested_object_size, get_default_complex_dtype
from ptychi.utils import generate_initial_opr_mode_weights

import test_utils as tutils

# THIS SHOULDN'T GO HERE, SHOULD SET THIS ELSEWHERE BUT FOR NOW I'M LEAVING IT
if os.environ.get('PTYCHO_CI_DATA_DIR') is None:
    os.environ["PTYCHO_CI_DATA_DIR"] = "/net/s8iddata/export/8-id-ECA/Analysis/atripath/ptychointerim-data/ci_data"
    
class Test2DPtychoRPIE_SDL(tutils.TungstenDataTester):
    @tutils.TungstenDataTester.wrap_recon_tester(name="test_2d_ptycho_rpie_synthesisdictlearn")
    def test_2d_ptycho_rpie_synthesisdictlearn(self):
        self.setup_ptychi(cpu_only=False)

        data, probe, pixel_size_m, positions_px = self.load_tungsten_data(additional_opr_modes=3)

        npz_dict_file = np.load(
            os.path.join(
                self.get_ci_input_data_dir(), "zernike2D_dictionaries", "Dlearned_orth.npz"
            )
        )
        D = npz_dict_file["D"]
        D_pinv = npz_dict_file["D_pinv"]
        npz_dict_file.close()

        options = api.RPIEOptions()
        options.data_options.data = data

        Nslices = 2
        options.object_options.initial_guess = torch.ones(
            [Nslices, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)],
            dtype=get_default_complex_dtype(),
        )
        options.object_options.pixel_size_m = pixel_size_m
        options.object_options.slice_spacings_m = (1e-5 / ( Nslices - 1)) * np.array( [1] * (Nslices - 1)).astype('float32')
        options.object_options.optimizable = True
        options.object_options.optimizer = api.Optimizers.SGD
        options.object_options.step_size = 1e-2
        options.object_options.alpha = 5e-1
        
        options.object_options.multislice_regularization.enabled = True
        options.object_options.multislice_regularization.weight = 0.01           
        options.object_options.multislice_regularization.unwrap_phase = True
        options.object_options.multislice_regularization.unwrap_image_grad_method = api.enums.ImageGradientMethods.FOURIER_DIFFERENTIATION
        options.object_options.multislice_regularization.unwrap_image_integration_method = api.enums.ImageIntegrationMethods.FOURIER
   
        options.probe_options.initial_guess = probe
        options.probe_options.optimizable = True
        options.probe_options.optimizer = api.Optimizers.SGD
        options.probe_options.orthogonalize_incoherent_modes.enabled = True
        options.probe_options.step_size = 1e-0
        options.probe_options.alpha = 9e-1

        options.probe_options.experimental.sdl_probe_options.enabled = True
        options.probe_options.experimental.sdl_probe_options.d_mat = np.asarray(
            D, dtype=np.complex64
        )
        options.probe_options.experimental.sdl_probe_options.d_mat_conj_transpose = np.conj(
            options.probe_options.experimental.sdl_probe_options.d_mat
        ).T
        options.probe_options.experimental.sdl_probe_options.d_mat_pinv = D_pinv
        options.probe_options.experimental.sdl_probe_options.probe_sparse_code_nnz = np.round(
            0.90 * D.shape[-1]
        )

        options.probe_position_options.position_x_px = positions_px[:, 1]
        options.probe_position_options.position_y_px = positions_px[:, 0]
        options.probe_position_options.optimizable = False

        options.opr_mode_weight_options.optimizable = True
        options.opr_mode_weight_options.initial_weights = generate_initial_opr_mode_weights( len(positions_px), probe.shape[0] )
        options.opr_mode_weight_options.optimization_plan.stride = 1
        options.opr_mode_weight_options.update_relaxation = 1e-2

        options.reconstructor_options.batch_size = round(data.shape[0] * 0.1)
        options.reconstructor_options.num_epochs = 50
        options.reconstructor_options.allow_nondeterministic_algorithms = False

        task = PtychographyTask(options)
        task.run()

        recon = task.get_data_to_cpu("object", as_numpy=True)[0]

        return recon


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-gold", action="store_true")
    args = parser.parse_args()

    tester = Test2DPtychoRPIE_SDL()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_2d_ptycho_rpie_synthesisdictlearn()
