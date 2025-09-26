import argparse

import torch
import numpy as np

from ptychi.data_structures.object import PlanarObject
from ptychi.api.options.base import ObjectOptions, RemoveGridArtifactsOptions
import ptychi.api as api

import test_utils as tutils


class TestRemoveGridArtifacts(tutils.BaseTester):

    def test_remove_grid_artifacts_phase(self):
        phase = torch.zeros([64, 64])
        phase[::10, ::10] = 1
        data = torch.ones([1, 64, 64]) * torch.exp(1j * phase)
        object = PlanarObject(
            data=data,
            options=ObjectOptions(
                pixel_size_m=1,
                remove_grid_artifacts=RemoveGridArtifactsOptions(
                    enabled=True,
                    component=api.MagPhaseComponents.PHASE,
                    period_x_m=10,
                    period_y_m=10,
                    window_size=3,
                    direction=api.Directions.XY,
                ),
            ),
        )
        with torch.no_grad():
            object.remove_grid_artifacts()
        
        data = object.data.detach().cpu().numpy()[0]

        if self.debug:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()
            ax.imshow(np.angle(data))
            plt.show()
        
        assert np.max(np.abs((np.angle(data)))) < 0.1
        
    def test_remove_grid_artifacts_phase_y(self):
        phase = torch.zeros([64, 64])
        phase[::10, ::10] = 1
        data = torch.ones([1, 64, 64]) * torch.exp(1j * phase)
        object = PlanarObject(
            data=data,
            options=ObjectOptions(
                pixel_size_m=1,
                remove_grid_artifacts=RemoveGridArtifactsOptions(
                    enabled=True,
                    component=api.MagPhaseComponents.PHASE,
                    period_y_m=10,
                    window_size=3,
                    direction=api.Directions.Y,
                ),
            ),
        )
        with torch.no_grad():
            object.remove_grid_artifacts()
        
        data = object.data.detach().cpu().numpy()[0]

        if self.debug:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()
            ax.imshow(np.angle(data))
            plt.show()
    
    def test_remove_grid_artifacts_both(self):
        phase = torch.zeros([64, 64])
        phase[::10, ::10] = 1
        data = torch.ones([1, 64, 64]) * torch.exp(1j * phase)
        object = PlanarObject(
            data=data,
            options=ObjectOptions(
                pixel_size_m=1,
                remove_grid_artifacts=RemoveGridArtifactsOptions(
                    enabled=True,
                    component=api.MagPhaseComponents.BOTH,
                    period_x_m=10,
                    period_y_m=10,
                    window_size=3,
                    direction=api.Directions.XY,
                ),
            ),
        )
        with torch.no_grad():
            object.remove_grid_artifacts()
        
        data = object.data.detach().cpu().numpy()[0]

        if self.debug:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()
            ax.imshow(np.angle(data))
            plt.show()
        
        assert np.max(np.abs((np.angle(data)))) < 0.1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()
    
    tester = TestRemoveGridArtifacts()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_remove_grid_artifacts_phase()
    tester.test_remove_grid_artifacts_phase_y()
    tester.test_remove_grid_artifacts_both()
