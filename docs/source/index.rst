.. pyRTC documentation master file, created by
   sphinx-quickstart on Tue May 14 10:01:34 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyRTC's documentation!
=================================

`pyRTC <https://github.com/jacotay7/pyRTC>`_ Adaptive optics (AO) is a technology that rapidly detects and corrects optical aberrations in order to significantly improve the resolution of an optical system. 
AO has been applied to imaging problems in the fields of astronomy, opthamology, and microscopy, as well as for various military and industrial applications.
AO systems operate using a so-called "real-time controller" or RTC, which is a catch all term used by the community to describe a set of software which is responsible for the conversion of optical aberration measurements to corresponding corrections.
For certain applications, particularly in astronomy, the speed of this software is crucial since the optical aberrations are highly dynamic.
In astronomical contexts, typical RTCs are required to control between 100-1000 degrees of freedom at speeds between 500-2000 Hz. 
These high performance constraints have traditionally restricted the use of slower, interpreted languages to lab environments where performance requirements are generally looser.
Moving towards high performance control for AO system in a language like python has the potential to bring new tools into the field of AO control while also making the field for accessible to the next generation of graduate students.

Github repository: https://github.com/jacotay7/pyRTC

Paper (Under Review): https://joss.theoj.org/papers/823c5feae7710aaf4ceb86adeda8a621

Main Features
--------------

pyRTC is an open-source, community-driven Python package for real-time control of AO systems, built with the following core goals:

- **Customizable High-Performance AO Pipeline:** Provide an efficient RTC pipeline with potential for full user customization.
- **Abstraction of Core AO System Components:** Facilitate support for a broad range of AO system architectures.
- **Open Library of API Examples:** Provide a library of examples for common hardware APIs used by the community to save time implementing basic hardware interactions.
- **Real-Time Monitoring and Interface Flexibility:** Support real-time access to intermediate data products, text-based user interaction, and straightforward integration with user-built GUIs.
- **Cross-Platform Compatibility:** Ensure broad usability across different operating systems.

.. toctree::
  :maxdepth: 1
  :caption: RTC Components

  components/wfs
  components/wfc
  components/slopes
  components/loop
  components/optimizer
  components/scicam
  components/modulator

.. toctree::
  :maxdepth: 1
  :caption: Guides

  guides/getting_started

.. toctree::
  :maxdepth: 1
  :caption: Examples

  examples/pywfs


Citing pyRTC
------------------------
To cite this project in publications:

TODO once JOSS paper is out.

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

pyRTC is an open-source software intended to be built for and by the AO community. Please use pyRTC freely and create new branches for your systems. 
We ask that if you implement new features that may be of broader interest to the community while integrating pyRTC into your AO system, please contribute to the code base by opening up an issue or pull request on github.   

Contact
------------

For feedback, collaboration, and feature requests you can contact me via e-mail at jacob.taylor@mail.utoronto.ca.
