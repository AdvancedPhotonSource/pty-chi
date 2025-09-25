Engines
=======

Pty-Chi supports multiple algorithms, or engines, for ptychography reconstruction. 
The table below compares their merits and limitations. 

.. list-table::
   :stub-columns: 1
   :widths: 40 40 40 40 40

   * - Engine
     - **LSQML**
     - **PIE (incl. ePIE, rPIE)**
     - **Difference map**
     - **Bilinear Hessian**
     - **Autodiff**
   * - Minibatching allowed
     - Yes
     - Yes  
     - No
     - Yes
     - Yes
   * - GPU support
     - Multiple
     - Single
     - Single
     - Single
     - Multiple
   * - Memory usage
     - Moderate
     - Moderate
     - High
     - Moderate
     - Low
   * - Mixed-state probe support
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - OPR support
     - Yes
     - Yes
     - No
     - No
     - Yes
   * - Multislice support
     - Yes
     - Yes
     - No
     - No
     - Yes

