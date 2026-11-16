Initialization Recommendations
==============================

`Ptychodus <https://github.com/AdvancedPhotonSource/ptychodus>`_ provides a series
of routines for generating the initial guesses and is recommended for general usage.
If you prefer generating the initial guesses manually, you can refer to the following
guidelines.


Object
------

The lateral size of the object should be the maximum extent of the probe positions
in x and y, plus the size of the probe, and plus a small margin to accommodate for
any probe position changes if position correction is enabled. You can get the
suggested lateral size using ``ptychi.utils.get_suggested_object_size``::

    h, w = get_suggested_object_size(probe_positions, probe_size, extra=50)


Initialize the object to be a complex array of 1s perturbed by a small amount of
Gaussian noise. For example::

    object = torch.ones([1, h, w], dtype=torch.complex64)
    object += torch.rand_like(object) * 1e-2


Note that the object should be a 3D tensor with the first dimension being the number
of slices. 


Multislice object
~~~~~~~~~~~~~~~~~

An object tensor with the length of the first dimension larger than 1 is automatically
interpreted as a multislice object. If the multislice object reconstructed is started
from scratch, you can use the similar random initialization as above. If you are initializing
the multislice object from a previous single-slice reconstruction, we recommend using
the n-slice-rooted magnitude for the initial guess's magnitude and its unwrapped phase
divided by the number of slices for the initial guess's phase::

    from ptychi.image_proc import unwrap_phase_2d

    # o1 is the single-slice object
    object = torch.abs(o1) ** (1 / n_slices) * torch.exp(1j * unwrap_phase_2d(o1) / n_slices)


Check the result of phase unwrapping before starting. You can try different settings of
``image_grad_method`` and ``image_integration_method`` in ``unwrap_phase_2d`` until the
result is satisfactory.


Probe
-----

The probe should be initialized to be a complex array of ``[n_opr_modes, n_incoherent_modes, h, w]``.

Squared probe magnitudes represent photons per pixel when the diffraction data
are in photon counts. All built-in propagators conserve their sum on the
propagating band. To match the initial probe power to the mean measured pattern,
assuming unit object transmission, use::

    probe = rescale_probe(probe, patterns)

If angular-spectrum propagation removes evanescent frequencies, pass a
propagator instance to include this filtering::

    probe = rescale_probe(probe, patterns, propagator=propagator)

The probe power constraint uses the same convention: ``probe_power`` is the
incident photon count per exposure. Under unit transmission and lossless
propagation onto the full detector grid, it equals the total detector count.
Divide squared field magnitudes by the pixel area of their plane to obtain
counts per unit area. Fresnel and Fraunhofer output pixel pitches are
``lambda * abs(z) / (W * dx)`` and ``lambda * abs(z) / (H * dy)``.


Incoherent probe modes
~~~~~~~~~~~~~~~~~~~~~~~~

If you only have a single incoherent mode and want to extend it to multiple modes, you can use
Hermite polynomials to generate the additional modes. See :func:`ptychi.utils.orthogonalize_initial_probe`.


OPR modes
~~~~~~~~~~

OPR modes can be initialized as random numbers normalized in such a way that the power of each
mode is the number of pixels in a mode (unit RMS amplitude).
See :func:`ptychi.utils.add_additional_opr_probe_modes_to_probe`.


Migrating saved probes
~~~~~~~~~~~~~~~~~~~~~~

To preserve predicted detector counts with an unchanged object, multiply probes
saved under the previous normalization by ``sqrt(H * W)`` for the default
Fourier propagator, or by ``sqrt(H * W) * abs(Fr / ar)`` for Fresnel and Fraunhofer
propagation. Here ``Fr = dx**2 / (lambda * z)`` and ``ar = dx / dy`` use the source
pixel pitches. Angular-spectrum probes retain their scale. Multiply saved
``probe_power`` targets by the square of the corresponding amplitude factor.
Apply the same amplitude factor to every stored OPR mode while retaining its
weights. Alternatively, initialize from measured data using ``rescale_probe``.

``FourierPropagator`` now uses fixed orthonormal normalization. For probes from
an explicitly selected legacy ``norm="ortho"``, the amplitude factor is 1;
for ``norm="forward"``, it is ``1 / sqrt(H * W)``. The normalization conversion
helpers are replaced by direct sums of photon counts.


OPR mode weights
-----------------

The OPR mode weights should be initialized to be 1 for the main OPR mode and small random numbers 
for the other modes. Usually, the secondary modes' weights are initialized to Gaussian random numbers
centered at 0 with a standard deviation of 1e-6. 
You can use :func:`ptychi.utils.generate_initial_opr_mode_weights` to generate
the initial OPR mode weights.
