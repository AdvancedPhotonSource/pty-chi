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

Probe synthesis uses a modified approximation of the reference algorithm's
weighted residual objective. Let :math:`S_k` be the native padded probe shift,
:math:`d_k=\widetilde P_k-S_kP` the local geometric-mean probe correction, and
:math:`w_k=|\widetilde O_k|^2` the local geometric-mean object power. The shared
probe correction is a weighted consensus of the individual adjoint corrections:

.. math::

   a_k &= S_k^*d_k, \\
   c_k &= \max\!\left(\operatorname{Re}(S_k^*w_k),\,
       0.01\,\operatorname{mean}_{\mathrm{pixels}}w_k\right), \\
   \Delta P &= \frac{\sum_{k\in B}c_k\odot a_k}{\sum_{k\in B}c_k},
       \qquad P_{\mathrm{new}}=P+\Delta P.

Multiplication, division, and the maximum are elementwise. Zero total confidence
uses the ordinary mean of the adjoint corrections. With finite intermediates,
a singleton minibatch gives exactly its direct-adjoint correction, including
a singleton at the end of a larger batch.

The original local normal operator is
:math:`H_k=S_k^*\operatorname{diag}(w_k)S_k`. Because the native shift preserves
constants, :math:`H_k\mathbf{1}=S_k^*w_k`: the confidence uses the real row sums
of this operator, projected onto the stated floor, rather than its diagonal.
The approximation follows from decomposing the local residual as

.. math::

   S_k\Delta P-d_k
   =S_k(\Delta P-a_k)+(S_kS_k^*-I)d_k.

It omits the linear term in the objective caused by the second term above and
replaces :math:`H_k` by :math:`\operatorname{diag}(c_k)`. The resulting quadratic
is centered at each adjoint correction, so its singleton update is preserved.
The 1% floor is a fixed stabilization choice. This is an approximate synthesis
rule; it does not claim exact minimization or monotone decrease of the original
weighted residual objective for the discrete padded shifts.

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
