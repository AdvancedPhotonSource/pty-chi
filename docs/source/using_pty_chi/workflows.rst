Workflows
=========

A :class:`~ptychi.api.task.PtychographyTask` represents one reconstruction run
with fixed initialization and settings. A workflow coordinates multiple tasks,
carrying reconstructed parameters from one task to the next to implement a
larger reconstruction strategy.


Multiscan shared-object reconstruction
--------------------------------------

:class:`~ptychi.workflows.MultiscanSharedObjectWorkflow` reconstructs several
datasets collected from sub-regions of one object. It creates one task per
dataset so that every scan retains its own probe, positions, OPR weights, and
optimizer state. After each task runs, only its reconstructed object tensor is
copied to the next task in a cyclic sequence.

Supply one options object and one data array per scan. Every supplied list must
have the same length, and at least one scan is required:

.. code-block:: python

    import ptychi.api as api
    from ptychi.workflows import MultiscanSharedObjectWorkflow

    workflow = MultiscanSharedObjectWorkflow(
        [task_options_1, task_options_2],
        diffraction_data=[diffraction_data_1, diffraction_data_2],
        object_data=[object_guess_1, object_guess_2],
        probe_data=[probe_guess_1, probe_guess_2],
        probe_position_x_px=[position_x_1, position_x_2],
        probe_position_y_px=[position_y_1, position_y_2],
        opr_mode_weights_data=[opr_weights_1, None],
        valid_pixel_mask=None,
        workflow_options=api.MultiscanSharedObjectWorkflowOptions(
            num_outer_epochs=20,
            num_inner_epochs=1,
        ),
    )
    workflow.run()

``opr_mode_weights_data`` and ``valid_pixel_mask`` may each be ``None`` for all
scans, or a list containing an array or ``None`` for each scan. The workflow
sets every task's progress total to ``num_outer_epochs * num_inner_epochs``;
the values originally stored in the caller's task options are not modified.

All scans must use the same object tensor shape, pixel geometry, and multislice
spacing. They must also use
``object_options.determine_position_origin_coords_by = ObjectPosOriginCoordsMethods.SUPPORT``.
The supplied positions must therefore already be expressed in a common,
SUPPORT-centered coordinate frame. The workflow does not align or stitch scan
coordinates.

Tasks are available in input order through ``workflow.tasks``. At completion,
the final object is copied to every task, while each task's probe, positions,
OPR weights, preconditioners, and optimizer state remain independent. Inactive
tasks are offloaded to CPU, and all tasks are on CPU when the workflow returns.
A workflow instance cannot be run a second time.


Progressive-resolution reconstruction
--------------------------------------

:class:`~ptychi.workflows.ProgressiveResolutionWorkflow` starts at reduced
spatial resolution and progressively increases the resolution until it reaches
the resolution of the supplied data. This can provide a useful coarse initial
solution for the subsequent, more expensive resolution levels.

The downsampling factor at level ``i`` is

.. math::

   2^{N - 1 - i},

where ``N`` is the total number of levels and ``i`` starts at zero. Thus, a
three-level workflow uses factors 4, 2, and 1.

At the first level, the workflow resizes the initial object and probe, divides
the probe positions by the factor, and increases the object pixel size by the
same factor. At every later level, it resizes the reconstructed object and
probe from the previous task, scales the reconstructed probe positions, and
copies the OPR mode weights. The final task therefore uses the full-resolution
data and the original object pixel size.


Basic usage
~~~~~~~~~~~

Configure the reconstruction algorithm as you would for a regular task, then
provide the number of resolution levels and the number of epochs at each
level:

.. code-block:: python

    import ptychi.api as api
    from ptychi.workflows import ProgressiveResolutionWorkflow

    task_options = api.LSQMLOptions()
    task_options.object_options.pixel_size_m = pixel_size_m
    task_options.object_options.optimizable = True
    task_options.object_options.optimizer = api.Optimizers.SGD
    task_options.object_options.step_size = 1
    task_options.probe_options.optimizable = True
    task_options.probe_options.optimizer = api.Optimizers.SGD
    task_options.probe_options.step_size = 1

    workflow_options = api.ProgressiveResolutionWorkflowOptions(
        num_resolution_levels=3,
        num_epochs_all_levels=[20, 20, 20],
    )

    workflow = ProgressiveResolutionWorkflow(
        task_options,
        diffraction_data=diffraction_data,
        object_data=object_guess,
        probe_data=probe_guess,
        probe_position_x_px=positions_px[:, 1],
        probe_position_y_px=positions_px[:, 0],
        opr_mode_weights_data=opr_mode_weights,
        valid_pixel_mask=valid_pixel_mask,
        workflow_options=workflow_options,
    )
    workflow.run()

``opr_mode_weights_data`` and ``valid_pixel_mask`` are optional, as they are
for :class:`~ptychi.api.task.PtychographyTask`. The workflow overrides
``task_options.reconstructor_options.num_epochs`` for each task using the
corresponding value in ``num_epochs_all_levels``. It does not modify the
original options object.


Far-field and near-field data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The workflow chooses how to reduce measured data from
``task_options.data_options.free_space_propagation_distance_m``:

* An infinite distance selects far-field ptychography. Diffraction patterns
  are reduced by cropping the low-frequency region in reciprocal space.
* A finite distance selects near-field ptychography. Measured intensity is
  real-space data, so it is resized in the same way as the object and probe.

For far-field data, set ``task_options.data_options.fft_shift`` according to
the layout of the supplied diffraction patterns. Set it to ``True`` when the
DC component is at the center; the dataset will FFT-shift the cropped pattern
to match the forward model, whose DC component is at the top-left corner. Set
it to ``False`` when the supplied data already has DC at the top-left corner.
For near-field data, this option should normally be ``False`` because the
measured intensity is already in real space.

When a validity mask is provided, the workflow crops it together with
far-field diffraction data. For near-field data it uses nearest-neighbor
resizing so that the mask remains boolean.


Accessing results from each level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every created task remains available in ``workflow.tasks``, ordered from the
lowest resolution to the full resolution. Use the normal task APIs to retrieve
reconstructed data:

.. code-block:: python

    for i_level, task in enumerate(workflow.tasks):
        object_at_level = task.get_data_to_cpu("object", as_numpy=True)
        probe_at_level = task.get_data_to_cpu("probe", as_numpy=True)

    full_resolution_task = workflow.get_full_resolution_task()
    reconstructed_object = full_resolution_task.get_data_to_cpu(
        "object", as_numpy=True
    )

:meth:`~ptychi.workflows.ProgressiveResolutionWorkflow.get_full_resolution_task`
is available only after all levels finish successfully. A workflow instance
cannot be run a second time.


Memory behavior
~~~~~~~~~~~~~~~

The workflow copies all constructor data to CPU memory. If a tensor supplied
by the caller is on an accelerator, the workflow emits a warning and creates a
CPU copy; it does not move or otherwise modify the caller's original tensor.
This means the caller remains responsible for releasing any original GPU
inputs.

After each resolution level finishes, the workflow calls
:meth:`~ptychi.api.task.PtychographyTask.set_large_tensor_device` to offload
the task's reconstruction parameters, diffraction patterns, optimizer state,
and registered reconstructor buffers to CPU memory. The cached task can still
be inspected through ``workflow.tasks`` without keeping every resolution
level resident on the GPU.


API reference
-------------

.. autoclass:: ptychi.workflows.MultiscanSharedObjectWorkflow
   :members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.workflow.MultiscanSharedObjectWorkflowOptions
   :members:
   :show-inheritance:

.. autoclass:: ptychi.workflows.ProgressiveResolutionWorkflow
   :members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.workflow.ProgressiveResolutionWorkflowOptions
   :members:
   :show-inheritance:
