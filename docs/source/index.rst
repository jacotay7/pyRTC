.. pyRTC documentation master file, created by
   sphinx-quickstart on Tue May 14 10:01:34 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyrtc's documentation!
=================================

`pyrtc <https://github.com/jacotay7/pyRTC>`_ is a Python toolkit for adaptive optics real-time control.
It is aimed at simulation-driven AO development, laboratory integration work, and controller research where a component-oriented Python stack is useful.

For the first stable release series:

- PyPI distribution name: ``pyrtcao``
- Python import name: ``pyRTC``
- Command-line prefix: ``pyrtc-*``
- Primary supported release surface: Linux on Python 3.9 through 3.13
- macOS and Windows currently have smoke-workflow coverage only

The package is organized around reusable AO components such as wavefront sensors, slope processors, loop controllers, wavefront correctors, science cameras, and telemetry producers.
These components can be composed in either a single-process development mode or a multi-process hardware-facing mode using shared-memory streams.

Github repository: https://github.com/jacotay7/pyRTC

Paper (Under Review): https://joss.theoj.org/papers/823c5feae7710aaf4ceb86adeda8a621

Main Features
--------------

pyRTC is an open-source, community-driven Python package for real-time control of AO systems, built with the following core goals:

- **Customizable High-Performance AO Pipeline:** Provide an efficient RTC pipeline with potential for full user customization.
- **Abstraction of Core AO System Components:** Facilitate support for a broad range of AO system architectures.
- **Open Library of API Examples:** Provide a library of examples for common hardware APIs used by the community to save time implementing basic hardware interactions.
- **Real-Time Monitoring and Interface Flexibility:** Support real-time access to intermediate data products, text-based user interaction, and straightforward integration with user-built GUIs.
- **Portable Development Workflow:** Keep the simulator and component model usable across environments while treating Linux as the primary operational target for `1.0.x`.

.. toctree::
  :maxdepth: 1
  :caption: RTC Components

  components/wfs
  components/wfc
  components/slopes
  components/loop
  components/optimizer
  components/scicam

.. toctree::
  :maxdepth: 1
  :caption: Guides

  guides/getting_started
  guides/architecture
  guides/developers_guide

.. toctree::
  :maxdepth: 1
  :caption: Examples

  examples/synthetic_shwfs
  examples/pywfs


Citing pyRTC
------------------------
To cite this project in publications:

The project paper is currently under review. Until the formal citation is finalized, cite the repository directly if your publication workflow requires an immediate reference.

.. .. code-block:: bibtex

..   @article{stable-baselines3,
..     author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
..     title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
..     journal = {Journal of Machine Learning Research},
..     year    = {2021},
..     volume  = {22},
..     number  = {268},
..     pages   = {1-8},
..     url     = {http://jmlr.org/papers/v22/20-1364.html}
..   }

Contributing
------------

Contributor and maintainer workflow guidance is collected in the Developer Guide.

Contact
------------

For feedback, collaboration, and feature requests you can contact me via e-mail at jtaylor@keck.hawaii.edu.
