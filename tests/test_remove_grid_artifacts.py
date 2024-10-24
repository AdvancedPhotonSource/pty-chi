import torch
import numpy as np

from ptychi.data_structures.object import Object2D
from ptychi.api.options.base import ObjectOptions
import ptychi.api as api


def test_remove_grid_artifacts(debug=False):
    phase = torch.zeros([64, 64])
    phase[::10, ::10] = 1
    data = torch.ones([64, 64]) * torch.exp(1j * phase)
    object = Object2D(
        data=data,
        options=ObjectOptions(
            pixel_size_m=1,
            remove_grid_artifacts=True,
            remove_grid_artifacts_period_x_m=10,
            remove_grid_artifacts_period_y_m=10,
            remove_grid_artifacts_window_size=3,
            remove_grid_artifacts_direction=api.Directions.XY,
        ),
    )
    with torch.no_grad():
        object.remove_grid_artifacts()
    
    data = object.data.detach().cpu().numpy()

    if debug:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        ax.imshow(np.angle(data))
        plt.show()
    
    assert np.max(np.abs((np.angle(data)))) < 0.1


if __name__ == "__main__":
    test_remove_grid_artifacts(debug=True)
