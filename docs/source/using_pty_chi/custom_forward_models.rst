Custom forward models and reconstruction parameters
===================================================

Pty-Chi's automatic differentiation (AD) reconstructors can be used with custom
forward models. This is useful when the measurement process includes effects
that are not described by the built-in model, or when additional quantities need
to be reconstructed along with the object and probe. PyTorch calculates the
gradients of the loss with respect to the reconstruction parameters, so you only
need to implement the forward calculation.

For ptychography, use ``AutodiffPtychographyReconstructor`` or its high-level
interface, ``api.AutodiffPtychographyOptions`` and ``PtychographyTask``. This
reconstructor includes ptychography-specific constraints and regularizers. For
other forward problems, you can use ``AutodiffReconstructor`` directly, as
described at the end of this page.


Using a custom forward model
----------------------------

Suppose the measured diffraction patterns contain a spatially uniform background.
The predicted intensity is then

.. math::

    I_i^{\mathrm{pred}}(q) = I_i^{\mathrm{ptycho}}(q) + b,

where :math:`b` is the background intensity per detector pixel, in the same units
as the measured data. In this example, the background is shared by all scan
positions. We first assume that its value is known.

You can extend ``PlanarPtychographyForwardModel`` to reuse the existing
ptychographic simulation and add the background to its output::

    import ptychi.api as api
    from ptychi.api.task import PtychographyTask
    from ptychi.forward_models import PlanarPtychographyForwardModel


    class BackgroundForwardModel(PlanarPtychographyForwardModel):
        def __init__(self, parameter_group, background=0.1, **kwargs):
            super().__init__(parameter_group, **kwargs)
            self.background = background

        def forward(self, indices):
            intensity = super().forward(indices)
            return intensity + self.background


``indices`` is a 1D tensor containing the scan indices in the current minibatch.
The returned tensor should have shape ``(batch_size, height, width)`` and contain
predicted intensities. The default loss function, ``MSE_SQRT``, takes square roots
internally. The built-in far-field model returns intensities with the DC component
at the top left corner, following the FFT convention. Any detector-dependent
correction should use the same pixel ordering.

Set ``forward_model_class`` to your class and put additional constructor arguments
in ``forward_model_params``. The reconstructor passes the parameter group as the
first argument. It also supplies the wavelength, detector size, propagation
distance, and shift settings, which are forwarded to the parent class through
``**kwargs`` above. For example::

    options = api.AutodiffPtychographyOptions()
    options.reconstructor_options.forward_model_class = BackgroundForwardModel
    options.reconstructor_options.forward_model_params = {"background": 0.1}
    options.reconstructor_options.num_epochs = 20
    options.reconstructor_options.batch_size = 16
    options.reconstructor_options.default_device = api.Devices.CPU

    options.object_options.pixel_size_m = pixel_size_m
    options.object_options.optimizer = api.Optimizers.ADAM
    options.object_options.step_size = 0.01
    options.probe_options.optimizer = api.Optimizers.ADAM
    options.probe_options.step_size = 0.01
    options.probe_position_options.optimizable = False
    options.opr_mode_weight_options.optimizable = False
    options.data_options.wavelength_m = wavelength_m

    task = PtychographyTask(
        options,
        diffraction_data=diffraction_data,
        object_data=object_guess,
        probe_data=probe_guess,
        probe_position_x_px=positions_px[:, 1],
        probe_position_y_px=positions_px[:, 0],
    )
    task.run()


Here, ``diffraction_data``, ``object_guess``, ``probe_guess``, ``positions_px``,
``pixel_size_m``, and ``wavelength_m`` are supplied from your data and initial
guesses. See :doc:`data_structures` for their shapes and
:doc:`initialization_recommendations` for initialization guidelines. This example
uses one OPR mode and a single CPU process. The default ``data_options.fft_shift``
assumes that the supplied far-field patterns have their DC component at the
center. Set it to ``False`` if the data already follow the FFT ordering.

The class itself is passed to ``forward_model_class``, rather than a constructed
instance. Values in ``forward_model_params``, such as ``background`` above, are
constructor settings. To optimize a quantity, add it to the parameter group as
described below.


Adding a reconstruction parameter
---------------------------------

Each reconstruction parameter in Pty-Chi is represented by a
``ReconstructParameter`` object. It stores the parameter tensor, its optimizer,
and its optimization schedule. A ``ParameterGroup`` collects these objects and
makes them available to the forward model and reconstructor.

To reconstruct the background, we can store the logarithm of its numerical
intensity value as a real-valued parameter and calculate :math:`b = \exp(\beta)`
in the forward model. This keeps the background positive during optimization.

First, extend the ptychography parameter group with an additional field::

    from dataclasses import dataclass
    import torch

    from ptychi.api.options.base import OptimizationPlan, ParameterOptions
    from ptychi.data_structures.base import ReconstructParameter
    from ptychi.data_structures.parameter_group import (
        PlanarPtychographyParameterGroup,
    )
    from ptychi.reconstructors.ad_ptychography import (
        AutodiffPtychographyReconstructor,
    )


    @dataclass
    class BackgroundParameterGroup(PlanarPtychographyParameterGroup):
        log_background: ReconstructParameter


The new forward model reads this parameter on each call::

    class LearnableBackgroundForwardModel(PlanarPtychographyForwardModel):
        def forward(self, indices):
            intensity = super().forward(indices)
            background = self.parameter_group.log_background.data.exp()
            return intensity + background


``ReconstructParameter.data`` returns a tensor connected to the computational
graph, including for complex parameters. Use PyTorch operations on this tensor
throughout the forward calculation. Calling ``detach()``, ``item()``, or converting
it to a NumPy array would break the gradient path. Detaching is appropriate when
fetching the result after reconstruction.

The high-level task normally builds a group containing the object, probe, probe
positions, and OPR mode weights. To include the background, override its
``build_reconstructor`` method. The following implementation is for a single
process::

    class BackgroundTask(PtychographyTask):
        def build_reconstructor(self):
            log_background = ReconstructParameter(
                data=torch.tensor([0.1]).log(),
                name="log_background",
                options=ParameterOptions(
                    optimizable=True,
                    optimizer=api.Optimizers.ADAM,
                    step_size=0.01,
                ),
            )
            parameter_group = BackgroundParameterGroup(
                object=self.object,
                probe=self.probe,
                probe_positions=self.probe_positions,
                opr_mode_weights=self.opr_mode_weights,
                log_background=log_background,
            )
            self.reconstructor = AutodiffPtychographyReconstructor(
                parameter_group=parameter_group,
                dataset=self.dataset,
                options=self.reconstructor_options,
            )
            self.reconstructor.build()


At this point in task construction, Pty-Chi has already configured the default
device and built the standard parameters and dataset. The new parameter is
initialized on that device with a background value of 0.1. Its ``ParameterOptions``
controls optimization in the same way as the options of the standard parameters.
For example, ``optimization_plan=OptimizationPlan(start=5)`` would start its
updates at epoch 5, with epochs counted from 0.

Use the same data and options as above, changing the forward model and task class::

    options.reconstructor_options.forward_model_class = LearnableBackgroundForwardModel
    options.reconstructor_options.forward_model_params = {}

    task = BackgroundTask(
        options,
        diffraction_data=diffraction_data,
        object_data=object_guess,
        probe_data=probe_guess,
        probe_position_x_px=positions_px[:, 1],
        probe_position_y_px=positions_px[:, 0],
    )
    task.run()

    background = task.reconstructor.parameter_group.log_background.data.exp()
    background = background.detach().cpu().item()
    print(background)


The base forward model registers optimizable parameters when it is constructed.
During reconstruction, the AD reconstructor differentiates the loss and steps
the optimizer of each enabled parameter in the group. This is why the additional
parameter must be included before the forward model is built. A standalone
``torch.nn.Parameter`` attached to the model would also receive gradients, but
Pty-Chi's optimizer loop operates on the ``ReconstructParameter`` objects in the
group.

Keep parameter-group fields for reconstruction parameters; fixed settings belong
in ``forward_model_params`` or on the model. For a fixed tensor used by a custom
model, use ``register_buffer`` so that it follows the model when devices change.
Custom results can be read from the parameter group as shown above;
``task.get_data`` provides access to the standard ptychography parameters.

The background in this example is a single scalar shared by all pixels and scan
positions. You can use a detector-sized tensor for a spatially varying background,
provided its pixel ordering agrees with the predicted intensity. In experimental
data, a flexible background can also absorb errors in the object or probe, so its
parameterization and initialization should reflect the measurement process.


Using the general AD reconstructor
----------------------------------

For a model that does not use the ptychographic simulation, inherit from
``ptychi.forward_models.ForwardModel`` and implement ``forward``. Call
``super().__init__(parameter_group)`` in your constructor so that the parameters
are registered. You can define a dataclass derived from ``ParameterGroup`` with
the ``ReconstructParameter`` fields needed by your model, using the same approach
as above.

Once you have constructed the parameter group and dataset, build and run the
general reconstructor directly::

    from ptychi.api.options.ad_general import AutodiffReconstructorOptions
    from ptychi.reconstructors.ad_general import AutodiffReconstructor

    reconstructor_options = AutodiffReconstructorOptions(
        forward_model_class=MyForwardModel,
        forward_model_params={},
        loss_function=api.LossFunctions.MSE,
        displayed_loss_function=api.LossFunctions.MSE,
        batch_size=16,
        num_epochs=20,
    )
    reconstructor = AutodiffReconstructor(
        parameter_group=parameter_group,
        dataset=dataset,
        options=reconstructor_options,
    )
    reconstructor.build()
    reconstructor.run()


``MyForwardModel``, ``parameter_group``, and ``dataset`` are your implementations
in this example. When working without ``PtychographyTask``, set the default device
and dtype before constructing tensors and parameters, as described in
:doc:`devices`. The general reconstructor passes only ``parameter_group`` and
``forward_model_params`` to the model constructor.

The dataset should implement ``__len__`` and ``__getitem__``, as well as
``move_attributes_to_device(device)``, which is called when the reconstructor is
built. Each sample should return ``(index, ..., measured_data)``. After batching,
the first element must be a 1D tensor of sample indices, and the last element is
the target passed to the loss function. All preceding elements are passed as
positional arguments to ``forward``. The current batch preparation moves these
inputs to the CPU; move them to the parameter device inside ``forward`` if they
are used in device-side calculations.

The general AD reconstructor handles the loss, differentiation, and optimizer
steps. Ptychography-specific constraints and regularizers belong to
``AutodiffPtychographyReconstructor``. If you need additional constraints for a
custom parameter, implement them in a reconstructor hook. When extending
ptychography hooks, call the parent implementation to retain the existing
behavior. Defining a ``post_update_hook`` on the new parameter alone does not
schedule it for execution.
