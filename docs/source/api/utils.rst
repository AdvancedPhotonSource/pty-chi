Utils
=====

Detector pixel masks
--------------------

:func:`ptychi.utils.get_constrained_pixel_mask` replaces
``IterativePtychographyReconstructor.get_constrained_pixel_mask``. Pass the
shared valid-pixel mask, measured-value threshold, and current measured batch:

.. code-block:: python

    from ptychi.utils import get_constrained_pixel_mask

    mask = get_constrained_pixel_mask(
        reconstructor.dataset.valid_pixel_mask,
        reconstructor.options.exclude_measured_pixels_below,
        y_true,
    )

Utils
-----

.. automodule:: ptychi.utils
    :members:
    :undoc-members:
    :show-inheritance:
