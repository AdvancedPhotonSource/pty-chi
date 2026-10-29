import argparse

import torch

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_suggested_object_size, get_default_complex_dtype

import test_utils as tutils


class Test2DPtychoBH(tutils.TungstenDataTester):

    @tutils.TungstenDataTester.wrap_recon_tester(name='test_2d_ptycho_bh_obj')
    def test_2d_ptycho_bh_obj(self):
        self.setup_ptychi(cpu_only=False)

        data, probe, pixel_size_m, positions_px = self.load_tungsten_data(additional_opr_modes=0)
        probe = probe[:, [0], :, :]

        options = api.BHOptions()
        diffraction_data = data

        object_data = torch.ones([1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], dtype=get_default_complex_dtype())
        options.object_options.pixel_size_m = pixel_size_m
        options.object_options.optimizable = True

        probe_data = probe
        options.probe_options.optimizable = False
        # Convert the legacy probe-coordinate weight to photon amplitudes.
        options.probe_options.rho = 0.1 * (probe.shape[-2] * probe.shape[-1]) ** 0.5

        probe_position_x_px = positions_px[:, 1]
        probe_position_y_px = positions_px[:, 0]
        options.probe_position_options.optimizable = False
        options.probe_position_options.rho = 2

        options.reconstructor_options.method = 'GD'
        options.reconstructor_options.batch_size = 96
        options.reconstructor_options.num_epochs = 16
        options.reconstructor_options.allow_nondeterministic_algorithms = False
        options.reconstructor_options.use_double_precision_for_fft = False
        options.reconstructor_options.forward_model_options.pad_for_shift = 4

        task = PtychographyTask(
            options,
            diffraction_data=diffraction_data,
            object_data=object_data,
            probe_data=probe_data,
            probe_position_x_px=probe_position_x_px,
            probe_position_y_px=probe_position_y_px,
        )
        task.run()

        recon = task.get_data_to_cpu('object', as_numpy=True)[0]

        return recon
    
    @tutils.TungstenDataTester.wrap_recon_tester(name='test_2d_ptycho_bh_obj_probe')
    def test_2d_ptycho_bh_obj_probe(self):
        self.setup_ptychi(cpu_only=False)

        data, probe, pixel_size_m, positions_px = self.load_tungsten_data(additional_opr_modes=0)
        probe = probe[:, [0], :, :]

        options = api.BHOptions()
        diffraction_data = data

        object_data = torch.ones([1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], dtype=get_default_complex_dtype())
        options.object_options.pixel_size_m = pixel_size_m
        options.object_options.optimizable = True

        probe_data = probe
        options.probe_options.optimizable = True
        # Convert the legacy probe-coordinate weight to photon amplitudes.
        options.probe_options.rho = 0.1 * (probe.shape[-2] * probe.shape[-1]) ** 0.5

        probe_position_x_px = positions_px[:, 1]
        probe_position_y_px = positions_px[:, 0]
        options.probe_position_options.optimizable = False
        options.probe_position_options.rho = 2

        options.reconstructor_options.method = 'GD'
        options.reconstructor_options.batch_size = 96
        options.reconstructor_options.num_epochs = 16
        options.reconstructor_options.allow_nondeterministic_algorithms = False
        options.reconstructor_options.use_double_precision_for_fft = False
        options.reconstructor_options.forward_model_options.pad_for_shift = 4

        task = PtychographyTask(
            options,
            diffraction_data=diffraction_data,
            object_data=object_data,
            probe_data=probe_data,
            probe_position_x_px=probe_position_x_px,
            probe_position_y_px=probe_position_y_px,
        )
        task.run()

        recon = task.get_data_to_cpu('object', as_numpy=True)[0]

        return recon 
    
    @tutils.TungstenDataTester.wrap_recon_tester(name='test_2d_ptycho_bh_obj_probe_pos')
    def test_2d_ptycho_bh_obj_probe_pos(self):
        self.setup_ptychi(cpu_only=False)

        data, probe, pixel_size_m, positions_px = self.load_tungsten_data(additional_opr_modes=0)
        probe = probe[:, [0], :, :]

        options = api.BHOptions()
        diffraction_data = data

        object_data = torch.ones([1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], dtype=get_default_complex_dtype())
        options.object_options.pixel_size_m = pixel_size_m
        options.object_options.optimizable = True

        probe_data = probe
        options.probe_options.optimizable = True
        # Convert the legacy probe-coordinate weight to photon amplitudes.
        options.probe_options.rho = 0.1 * (probe.shape[-2] * probe.shape[-1]) ** 0.5

        probe_position_x_px = positions_px[:, 1]
        probe_position_y_px = positions_px[:, 0]
        options.probe_position_options.optimizable = True
        options.probe_position_options.rho = 2

        options.reconstructor_options.method = 'GD'
        options.reconstructor_options.batch_size = 96
        options.reconstructor_options.num_epochs = 16
        options.reconstructor_options.allow_nondeterministic_algorithms = False
        options.reconstructor_options.use_double_precision_for_fft = False
        options.reconstructor_options.forward_model_options.pad_for_shift = 4

        task = PtychographyTask(
            options,
            diffraction_data=diffraction_data,
            object_data=object_data,
            probe_data=probe_data,
            probe_position_x_px=probe_position_x_px,
            probe_position_y_px=probe_position_y_px,
        )
        task.run()

        recon = task.get_data_to_cpu('object', as_numpy=True)[0]

        return recon 
    
    @tutils.TungstenDataTester.wrap_recon_tester(name='test_2d_ptycho_bh_obj_probe_pos_cg')
    def test_2d_ptycho_bh_obj_probe_pos_cg(self):
        self.setup_ptychi(cpu_only=False)

        data, probe, pixel_size_m, positions_px = self.load_tungsten_data(additional_opr_modes=0)
        probe = probe[:, [0], :, :]

        options = api.BHOptions()
        diffraction_data = data

        object_data = torch.ones([1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], dtype=get_default_complex_dtype())
        options.object_options.pixel_size_m = pixel_size_m
        options.object_options.optimizable = True

        probe_data = probe
        options.probe_options.optimizable = True
        # Convert the legacy probe-coordinate weight to photon amplitudes.
        options.probe_options.rho = 0.1 * (probe.shape[-2] * probe.shape[-1]) ** 0.5

        probe_position_x_px = positions_px[:, 1]
        probe_position_y_px = positions_px[:, 0]
        options.probe_position_options.optimizable = True
        options.probe_position_options.rho = 2

        options.reconstructor_options.method = 'CG'
        options.reconstructor_options.batch_size = 96
        options.reconstructor_options.num_epochs = 8
        options.reconstructor_options.allow_nondeterministic_algorithms = False
        options.reconstructor_options.use_double_precision_for_fft = False
        options.reconstructor_options.forward_model_options.pad_for_shift = 4

        task = PtychographyTask(
            options,
            diffraction_data=diffraction_data,
            object_data=object_data,
            probe_data=probe_data,
            probe_position_x_px=probe_position_x_px,
            probe_position_y_px=probe_position_y_px,
        )
        task.run()

        recon = task.get_data_to_cpu('object', as_numpy=True)[0]

        return recon 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = Test2DPtychoBH()
    tester.setup_method(name="", generate_data=False,
                        generate_gold=args.generate_gold, debug=True)
    tester.test_2d_ptycho_bh_obj()
    tester.test_2d_ptycho_bh_obj_probe()
    tester.test_2d_ptycho_bh_obj_probe_pos()
    tester.test_2d_ptycho_bh_obj_probe_pos_cg()
    