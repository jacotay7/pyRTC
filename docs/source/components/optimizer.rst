.. wfs:

.. currentmodule:: pyRTC.Optimizer


Optimizer
=========

The `Optimizer` component provides a general framework for closed-loop or supervisory optimization tasks in `pyRTC`.
It is intended for workflows where a control parameter or system configuration needs to be tuned over repeated trials.

Internally, the base class uses Optuna with a CMA-ES sampler and is designed to be subclassed for concrete optimization tasks such as NCPA tuning or loop hyperparameter search.

Soft-RTC Example
----------------

The following example shows the general `soft-RTC` pattern for an optimizer object.
In practice, users usually subclass `Optimizer` and implement task-specific objective and application methods.

.. code-block:: python

  from pyRTC.Optimizer import Optimizer

  class MyOptimizer(Optimizer):
      def objective(self, trial):
          # Evaluate the current system state and return a scalar score.
          return 0.0

      def applyTrial(self, trial):
          # Push the candidate parameters into the system.
          return

      def applyOptimum(self):
          # Apply the best known parameters once optimization is done.
          return

  optimizer = MyOptimizer({"numSteps": 20, "functions": []})
  optimizer.optimize()

Hard-RTC Example
----------------

For hardware-facing or supervisory workflows, the optimizer can also run in a separate process.

.. code-block:: python
  
  from pyRTC.Pipeline import hardwareLauncher

  config = 'path/to/config.yaml'
  port = 3006

  optimizer = hardwareLauncher('path/to/pyRTC/hardware/myOptimizer.py', config, port)
  optimizer.launch()
  optimizer.run("optimize")

Implementation Notes
--------------------

The base class is intentionally generic. A real optimizer subclass usually defines:

- how a trial is applied to the running system
- how the objective value is measured
- how the best result is committed once optimization ends

The examples under `pyRTC.hardware` are the right starting point for system-specific tuning logic.


Parameters
----------

.. autoclass:: Optimizer
  :members:
  :inherited-members:
  :no-index:
