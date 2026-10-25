Options
=======

The ``Options`` dataclasses are the main user-facing interface for configuring
Pty-Chi when you use the TaskManager API. In a typical workflow, you build one
high-level options object, fill in its fields, and pass it to
``ptychi.api.task.PtychographyTask``.


Hierarchy of Options Objects
----------------------------

The highest-level options classes used for ptychography tasks are subclasses of
``PtychographyTaskOptions``. These classes bundle together the configuration of
the full reconstruction job:

- ``data_options``: diffraction patterns and experimental metadata
- ``reconstructor_options``: settings for the reconstruction loop, such as
  batch size and number of epochs
- ``object_options``: settings for the reconstructed object
- ``probe_options``: settings for the reconstructed probe
- ``probe_position_options``: settings for scan positions and position correction
- ``opr_mode_weight_options``: settings for OPR mode weights

The top-level subclass determines the reconstruction engine and the concrete
types of the nested options objects. For example:

.. code-block:: python

    import ptychi.api as api

    options = api.LSQMLOptions()
    # Uses LSQMLReconstructorOptions, LSQMLObjectOptions, LSQMLProbeOptions, ...

    options = api.EPIEOptions()
    # Uses EPIE / PIE-specific nested options

    options = api.AutodiffPtychographyOptions()
    # Uses autodiff-specific nested options

In other words, choosing ``api.LSQMLOptions()``, ``api.PIEOptions()``,
``api.DMOptions()``, ``api.RAAROptions()``, or another top-level options class
is how you choose the engine to run.


Constructing an Options Object
------------------------------

The example below shows the typical pattern: instantiate a top-level task
options object, then populate its nested options objects.

.. code-block:: python

    import torch
    import ptychi.api as api
    from ptychi.utils import get_default_complex_dtype, get_suggested_object_size

    data, probe, pixel_size_m, positions_px = your_data_loading_function()

    options = api.LSQMLOptions()

    object_guess = torch.ones(
        [1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)],
        dtype=get_default_complex_dtype(),
    )
    options.object_options.pixel_size_m = pixel_size_m
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = 1.0

    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = 1.0

    options.probe_position_options.optimizable = False

    options.reconstructor_options.batch_size = 64
    options.reconstructor_options.num_epochs = 16

After the options object is configured, pass it to ``PtychographyTask``:

.. code-block:: python

    from ptychi.api.task import PtychographyTask

    task = PtychographyTask(
        options,
        diffraction_data=data,
        object_data=object_guess,
        probe_data=probe,
        probe_position_x_px=positions_px[:, 1],
        probe_position_y_px=positions_px[:, 0],
    )
    task.run()

Large reconstruction arrays are task data, not settings. During the transition,
the old option fields such as ``options.data_options.data`` and
``options.probe_options.initial_guess`` still work, but they emit
``DeprecationWarning`` when ``PtychographyTask`` resolves them.


Object Shift Operators
----------------------

The forward model automatically selects its object sampling operator through
``reconstructor_options.forward_model_options.shift_operator``:

- ``api.ShiftOperators.AUTO`` (default) selects full-object shifting when the
  free-space propagation distance is finite and the probe height and width each
  strictly exceed 80% of the corresponding object dimensions on the probe pixel
  grid. Different pixel sizes and aspect ratios are included in this comparison.
- ``api.ShiftOperators.PATCH`` selects the existing patch extraction and placement
  operators.
- ``api.ShiftOperators.FULL_OBJECT`` selects full-object shifting for any geometry.

For example, to force full-object sampling:

.. code-block:: python

    options.reconstructor_options.forward_model_options.shift_operator = (
        api.ShiftOperators.FULL_OBJECT
    )
    options.object_options.patch_interpolation_method = api.PatchInterpolationMethods.FOURIER

Direct construction of ``PlanarPtychographyForwardModel`` accepts the same
``shift_operator`` argument. Its ``resolved_shift_operator`` property reports the
current selection, including after geometry changes; the requested option remains
unchanged.

Full-object sampling shifts each complete object slice periodically, then uses a
fixed center crop or zero-padding to match the probe shape. For a probe larger
than the object, pixels outside the shifted object support are zero. Features
crossing an object boundary wrap to the opposite side. Choose the object support
with this periodic boundary model in mind. The ``pad_for_shift`` option applies
to patch sampling; full-object sampling uses the original object grid.

The ``patch_interpolation_method`` setting selects Fourier, bilinear, or nearest
interpolation in either branch. Full-object bilinear and nearest interpolation
use spectral representations of circular interpolation kernels. All three have
an exact additive adjoint. Full-object displacement is applied entirely to the
object, and the shared probe-shift helpers act as identity operations.

Position derivatives are evaluated on the complete object before cropping.
Bilinear derivatives are piecewise defined, and nearest interpolation has zero
position derivatives almost everywhere; use Fourier or bilinear interpolation
for gradient-based position refinement. Cross-correlation position correction
remains available where supported by the reconstructor.

DM and RAAR retain normalized-adjoint object updates. These are approximate
projections for fractional full-object interpolation, because the normal
operator couples pixels. Their full-object normalization uses a denominator
floor of 5% of its maximum to stabilize weakly illuminated pixels. Convergence
can differ from patch sampling, particularly with cropped, nonuniform
illumination.


ParameterOptions
----------------

The object, probe, probe positions, and OPR mode weights are each configured by
a subclass of ``ParameterOptions``. These parameter-level options define:

- whether the parameter is optimizable
- which optimizer to use
- the initial ``step_size``
- optimizer keyword arguments in ``optimizer_params``
- when the parameter is optimized through ``optimization_plan``

For example, you can optimize the object from the beginning of the run, but
delay probe updates until later:

.. code-block:: python

    import ptychi.api as api

    options = api.LSQMLOptions()

    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = 1.0

    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = 1.0
    options.probe_options.optimization_plan.start = 5


Independent object and probe pixel sizes
----------------------------------------

The object and probe may use different real-space sampling.  Object pixels are
configured by ``object_options.pixel_size_m`` (the x/width dimension) and
``object_options.pixel_size_aspect_ratio`` (width divided by height).  The
corresponding optional probe settings are
``probe_options.pixel_size_m`` and
``probe_options.pixel_size_aspect_ratio``.  Each omitted probe value inherits
the matching object value independently.  For example, this changes only the
probe's y sampling:

.. code-block:: python

    options.probe_options.pixel_size_m = None
    options.probe_options.pixel_size_aspect_ratio = 1.25

Pixel height is ``pixel_size_m / pixel_size_aspect_ratio``.  For each axis, a
native object dimension ``n`` is represented internally on the probe grid with

.. math::

    n_{probe} = \max(1, \operatorname{round}(n\,\Delta_{object}/\Delta_{probe})).

Pty-Chi resamples the complete multislice object by centered Fourier
padding/cropping, using constant-amplitude normalization.  Analytical object
gradients are mapped back with the matching Hermitian adjoint; direct object
projections use its normalized pseudoinverse.  The equal-pixel-size case keeps
the original no-resampling path.


OptimizationPlan
----------------

Each ``ParameterOptions`` object contains an ``optimization_plan``. This plan
controls when the corresponding parameter is updated.

The most commonly used fields are:

- ``start``: first epoch where optimization is allowed
- ``stop``: first epoch where optimization is no longer allowed
- ``stride``: optimize every ``stride`` epochs
- ``step_size_scheduler_class``: optional scheduler class name from
  ``torch.optim.lr_scheduler``
- ``step_size_scheduler_options``: keyword arguments for that scheduler,
  excluding ``optimizer``

Example:

.. code-block:: python

    import ptychi.api as api

    options = api.EPIEOptions()

    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = 0.1

    plan = options.probe_options.optimization_plan
    plan.start = 10
    plan.stop = 100
    plan.stride = 2

This means the probe starts updating at epoch 10, stops before epoch 100, and
is updated every other epoch.


Step Size Schedulers
--------------------

Step size schedulers are configured per parameter through
``ParameterOptions.optimization_plan``. This is useful when the object, probe,
and probe positions need different schedules.

To enable a scheduler, set ``step_size_scheduler_class`` to the name of a class
in ``torch.optim.lr_scheduler`` and pass the scheduler constructor arguments in
``step_size_scheduler_options``.

.. code-block:: python

    import ptychi.api as api

    options = api.LSQMLOptions()

    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = 1.0
    options.object_options.optimization_plan.step_size_scheduler_class = "ExponentialLR"
    options.object_options.optimization_plan.step_size_scheduler_options = {
        "gamma": 0.98,
    }

    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.Adam
    options.probe_options.step_size = 1e-2
    options.probe_options.optimization_plan.step_size_scheduler_class = "StepLR"
    options.probe_options.optimization_plan.step_size_scheduler_options = {
        "step_size": 20,
        "gamma": 0.5,
    }

If ``step_size_scheduler_class`` is left as ``None``, no scheduler is used for
that parameter.

Schedulers are stepped internally at the end of each epoch while the parameter
is inside its optimization interval, so the same scheduler mechanism applies to
parameters updated through ``optimizer.step()`` and to parameters whose update
rules use the current step size explicitly.
