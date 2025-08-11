Getting Started
===============

Installation
------------

Standard installation
~~~~~~~~~~~~~~~~~~~~~
The easiest way to install the latest release is through PyPI. 

First, create a new conda environment with Python 3.11:
::

    conda create -n ptychi python=3.11

Then install Pty-Chi using::

    pip install ptychi


Developer installation
~~~~~~~~~~~~~~~~~~~~~~

Use developer installation when you want to modify the code and test the changes,
or when you run into build issues that drive you to install the package from source.
We recommend using uv for environment and package management.

Option 1: Installation with uv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Uv is a modern lightweight package manager for Python featuring fast speed and
deterministic builds. When creating a uv virtual environment, the environment
directory and all the packages inatalled in it are kept in the current working
directory -- unlike Conda, where the environments are centrally managed. Therefore,
first ``cd`` into the **root level** of your local clone of Pty-Chi, and then create
a new uv virtual environment with Python 3.11::

    uv venv --python 3.11

Activate the environment::

    source .venv/bin/activate

Then install Pty-Chi and its dependencies using::

    uv pip install -r requirements.txt
    uv pip install -e .

Option 2: Installation with Conda and pip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install the latest code in the `main` branch, clone the repository to your workspace, and create a new conda environment
using::

    conda create -n ptychi python=3.11

Then install the package and its dependencies using::

    pip install -r requirements.txt
    pip install -e .


Basic Usage
-----------

Using Pty-Chi with Ptychodus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ptychodus provides an easy-to-use graphical interface for running Pty-Chi.
Get Ptychodus from `here <https://github.com/AdvancedPhotonSource/ptychodus>`_.

Using Pty-Chi with its Python API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also run reconstructions with Pty-Chi's Python API. Below is an example
of how to run a reconstruction with the least-squares maximum likelihood (LSQML)
algorithm. A few key points:

- Select the algorithm by calling the appropriate ``Options`` class. For example, by
  calling ``api.LSQMLOptions()``, you are selecting the LSQML algorithm.
- Data options, object options, probe positions, probe position options, and 
  reconstructor options are mandated.
- Pty-Chi does not generate initial guesses for the object, probe, or OPR mode weights.
  You can either generate them yourself, or create the initial guesses using Ptychodus.
- OPR mode weight options are optional and is used when multiple orthogonal probe relaxation 
  (OPR) modes are used.
- Use enumerations to set multiple-choice options, like ``api.Optimizers.SGD``.


.. code-block:: python

    import torch
    import ptychi.api as api
    from ptychi.api.task import PtychographyTask
    from ptychi.utils import get_suggested_object_size, get_default_complex_dtype, generate_initial_opr_mode_weights


    torch.set_default_device("cuda")  # Important: set default device to enable GPU
    torch.set_default_dtype(torch.float32)
    set_default_complex_dtype(torch.complex64)

    data, probe, pixel_size_m, positions_px = your_data_loading_function()

    # Create options
    options = api.LSQMLOptions()
    
    options.data_options.data = your_diffraction_data
    
    options.object_options.initial_guess = torch.ones(
        [1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)],
        dtype=get_default_complex_dtype()
    )
    options.object_options.pixel_size_m = pixel_size_m
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = 1
    
    options.probe_options.initial_guess = probe
    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = 1

    options.probe_position_options.position_x_px = positions_px[:, 1]
    options.probe_position_options.position_y_px = positions_px[:, 0]
    options.probe_position_options.optimizable = False
    
    options.opr_mode_weight_options.initial_weights = generate_initial_opr_mode_weights(len(positions_px), probe.shape[0])
    options.opr_mode_weight_options.optimizable = True
    options.opr_mode_weight_options.update_relaxation = 0.1
    
    options.reconstructor_options.batch_size = 44
    options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
    options.reconstructor_options.num_epochs = 8
    
    # Run reconstruction
    task = PtychographyTask(options)
    task.run()
    
    # To get and save results after every ``save_interval`` epochs, you can also do:
    # for epoch in range(0, options.reconstructor_options.num_epochs, save_interval):
    #     task.run(save_interval)
    #     recon = task.get_data_to_cpu('object', as_numpy=True)[0]
    #     np.save(recon, f"recon_epoch_{epoch}.npy")

    recon = task.get_data_to_cpu('object', as_numpy=True)[0]

    # Or use
    # recon = task.object.get_object_in_roi().cpu().numpy()
    # To get the reconstructed object within the ROI.

Pty-Chi's repository contains a collection of testers that also serve as example scripts.
You can find them in the `test/` directory.