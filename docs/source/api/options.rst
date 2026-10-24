Options
=======

The following classes define the configuration options for Pty-chi reconstructions.

Base Options
------------

.. autoclass:: ptychi.api.options.base.Options
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.TaskOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.ObjectOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.ProbeOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.ProbePositionOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.OPRModeWeightsOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.FeatureOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.SynthesisDictLearnProbeOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.PositionCorrectionOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.PositionAffineTransformConstraintOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.ForwardModelOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.ObjectMultisliceRegularizationOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.ObjectL1NormConstraintOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.ObjectL2NormConstraintOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.ObjectSmoothnessConstraintOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.ObjectTotalVariationOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.RemoveGridArtifactsOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.RemoveObjectProbeAmbiguityOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.SliceSpacingOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.ProbePowerConstraintOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.ProbeOrthogonalizeIncoherentModesOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.ProbeOrthogonalizeOPRModesOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.ProbeSupportConstraintOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.ProbeCenterConstraintOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.base.OPRModeWeightsSmoothingOptions
   :members:
   :undoc-members:
   :show-inheritance:

Data Options
------------

.. autoclass:: ptychi.api.options.data.PtychographyDataOptions
   :members:
   :undoc-members:
   :show-inheritance:

Difference Map (DM) Options
---------------------------

.. autoclass:: ptychi.api.options.dm.DMOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.dm.DMReconstructorOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.dm.DMObjectOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.dm.DMProbeOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.dm.DMProbePositionOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.dm.DMOPRModeWeightsOptions
   :members:
   :undoc-members:
   :show-inheritance:

Relaxed Averaged Alternating Reflections (RAAR) Options
-------------------------------------------------------

.. autoclass:: ptychi.api.options.raar.RAAROptions
   :members:
   :undoc-members:
   :show-inheritance:

Reconstructor Options
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ptychi.api.options.raar.RAARReconstructorOptions
   :members:
   :undoc-members:
   :show-inheritance:

Object Options
~~~~~~~~~~~~~~
.. autoclass:: ptychi.api.options.raar.RAARObjectOptions
   :members:
   :undoc-members:
   :show-inheritance:

Probe Options
~~~~~~~~~~~~~
.. autoclass:: ptychi.api.options.raar.RAARProbeOptions
   :members:
   :undoc-members:
   :show-inheritance:

Probe Position Options
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ptychi.api.options.raar.RAARProbePositionOptions
   :members:
   :undoc-members:
   :show-inheritance:

OPR Mode Weight Options
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ptychi.api.options.raar.RAAROPRModeWeightsOptions
   :members:
   :undoc-members:
   :show-inheritance:

PIE Options
-----------

.. autoclass:: ptychi.api.options.pie.PIEOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.pie.PIEReconstructorOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.pie.PIEObjectOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.pie.PIEProbeOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.pie.PIEProbePositionOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.pie.PIEOPRModeWeightsOptions
   :members:
   :undoc-members:
   :show-inheritance:

GM-rPIE and GM-MAGPIE Options
----------------------------

Both algorithms use ``MAGPIEOptions``. Set
``reconstructor_options.multigrid_levels = 1`` for GM-rPIE or a larger
valid depth for GM-MAGPIE. The initial implementation supports one object
slice and one coherent shared probe, random minibatches, and matching
object/probe/detector sampling. Object and probe must both be optimized
on every minibatch, with unit SGD settings and no step-size schedulers.
Native gradient position correction and post-epoch constraints remain available.

.. code-block:: python

   from ptychi.api import enums
   from ptychi.api.options import MAGPIEOptions
   from ptychi.api.task import PtychographyTask

   options = MAGPIEOptions()
   options.reconstructor_options.default_device = enums.Devices.CPU
   options.reconstructor_options.multigrid_levels = 1  # GM-rPIE
   # Supply your measured intensities, initial guesses, and positions.
   task = PtychographyTask(
       options,
       diffraction_data=intensities,
       object_data=object_guess,
       probe_data=probe_guess,
       probe_position_x_px=positions_x,
       probe_position_y_px=positions_y,
   )
   task.run()

Probe synthesis approximates the reference algorithm's weighted residual in
common probe coordinates. Let :math:`S_k` be the native padded probe shift,
:math:`d_k=\widetilde P_k-S_kP` the local geometric-mean probe correction, and
:math:`\widetilde O_k` the local geometric-mean object patch. Define :math:`T_k`
as the native coordinate shift with the opposite fractional displacement,
using the same interpolation, replicate padding, and cropping. It differs
from the actual padded adjoint :math:`S_k^*`.

The shared probe correction minimizes a nonnegative weighted consensus:

.. math::

   a_k &= S_k^*d_k, \\
   c_k &= |T_k\widetilde O_k|^2, \\
   \Delta P &= \underset{\delta}{\operatorname{argmin}}\;
       \frac12\sum_{k\in B}\left\|
       (T_k\widetilde O_k)\odot(\delta-a_k)\right\|_2^2
       = \frac{\sum_{k\in B}c_k\odot a_k}{\sum_{k\in B}c_k}, \\
   P_{\mathrm{new}} &= P+\Delta P.

Multiplication and division are elementwise; the displayed ratio applies where
the total confidence is positive. The confidence has no floor or clipping.
Zero total confidence uses the ordinary mean of the adjoint
corrections. With finite intermediates, a singleton minibatch gives exactly
its direct-adjoint correction, including a singleton at the end of a larger
batch.

For ideal coordinate translations on a full or periodic domain,
:math:`T_k=S_k^{-1}=S_k^*`, and translation preserves both the norm and pointwise
products. Consequently,

.. math::

   \left\|\widetilde O_k\odot(S_k\delta-d_k)\right\|_2^2
   =\left\|(T_k\widetilde O_k)\odot(\delta-S_k^*d_k)\right\|_2^2.

Discrete interpolation does not generally preserve products, and padding makes
the reverse coordinate shift different from the adjoint. The implemented
consensus is therefore an approximation; it does not guarantee exact
minimization or monotone decrease of the original weighted residual objective.

.. autoclass:: ptychi.api.options.magpie.MAGPIEOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.magpie.MAGPIEReconstructorOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.magpie.MAGPIEObjectOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.magpie.MAGPIEProbeOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.magpie.MAGPIEProbePositionOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.magpie.MAGPIEOPRModeWeightsOptions
   :members:
   :undoc-members:
   :show-inheritance:

LSQML Options
-------------

.. autoclass:: ptychi.api.options.lsqml.LSQMLOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.lsqml.LSQMLReconstructorOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.lsqml.LSQMLObjectOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.lsqml.LSQMLProbeOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.lsqml.LSQMLProbePositionOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.lsqml.LSQMLOPRModeWeightsOptions
   :members:
   :undoc-members:
   :show-inheritance:

Autodiff Ptychography Options
-----------------------------

.. autoclass:: ptychi.api.options.ad_ptychography.AutodiffPtychographyOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.ad_ptychography.AutodiffPtychographyReconstructorOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.ad_ptychography.AutodiffPtychographyObjectOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.ad_ptychography.AutodiffPtychographyProbeOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.ad_ptychography.AutodiffPtychographyProbePositionOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ptychi.api.options.ad_ptychography.AutodiffPtychographyOPRModeWeightsOptions
   :members:
   :undoc-members:
   :show-inheritance:
