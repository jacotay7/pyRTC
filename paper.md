---
title: 'pyRTC: An open-source Python solution for kHz real-time control of adaptive optics systems'
tags:
  - Python
  - astronomy
  - adaptive optics
  - real-time control
authors:
  - name: Jacob Taylor
    orcid: 0000-0002-6356-567X
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Robin Swanson
    affiliation: "2, 3"
  - name: Suresh Sivanandam
    affiliation: "1, 2"
affiliations:
 - name: David A. Dunlap Department of Astronomy and Astrophysics, University of Toronto, 50 St George St, Toronto, ON M5S 3H4, Canada
   index: 1
 - name: Dunlap Institute for Astronomy and Astrophysics, University of Toronto, 50 St George St, Toronto, ON M5S 3H4, Canada
   index: 2
 - name: Department of Computer Science, University of Toronto, 40 St George St, Toronto, ON M5S 2E4, Canada
   index: 3
date: 10 January 2024
bibliography: paper.bib
---
## Summary

Adaptive optics (AO) is a technology that rapidly detects and corrects optical aberrations to significantly improve the resolution of an optical system. AO has been applied to imaging problems in fields such as astronomy, ophthalmology, and microscopy, as well as various military and industrial applications. AO systems operate using a so-called `real-time controller' (RTC), a term used by the community to describe software responsible for converting optical aberration measurements into corresponding corrections. In astronomical contexts, RTCs typically control 100-1000 degrees of freedom at speeds between 500-2000 Hz.

pyRTC is an open-source, community-driven Python package for real-time control of AO systems, built with the following core goals:

- **Customizable High-Performance AO Pipeline:** Provide an efficient RTC pipeline with potential for full user customization.
- **Abstraction of Core AO System Components:** Facilitate support for a broad range of AO system architectures.
- **Open Library of API Examples:** Provide a library of examples for common hardware APIs used by the community to save time implementing basic hardware interactions.
- **Real-Time Monitoring and Interface Flexibility:** Support real-time access to intermediate data products, text-based user interaction, and straightforward integration with user-built GUIs.
- **Cross-Platform Compatibility:** Ensure broad usability across different operating systems.

In this publication, we present a pre-alpha version of the pyRTC package to the community and invite them to try it out on their hardware, provide feedback, and contribute to the code base or hardware API library.

## Statement of Need

Hardware providers for AO system components (cameras, deformable mirrors, etc...) currently provide API support for only three programming languages: C/C++, Python, and MATLAB. High-performance RTCs have been developed in C/C++ (e.g., CACAO [`@CACAO`], DAO, DARC[`@DARC`], HEART[`@HEART`]), while off-the-shelf MATLAB controllers are available for purchase. Off-the-shelf RTC solutions can be costly, and they lack customizability and transparency. The community-led C++ solutions, known for their performance, can be complex to understand and implement, leaving AO researchers with limited options: expensive RTCs, investing in software expertise for C++ solutions, or creating custom low-performance RTCs.

pyRTC is an open-source RTC software for AO that aims to be the highest performance free AO control software available in Python, maintaining sufficient user-friendliness for the average AO researcher. pyRTC abstracts the variable hardware components in an AO system into high-level control objects and provides an architecture for combining those objects into a high-performance pipeline. Traditional performance limitations in Python, due to the Global Interpreter Lock (GIL), are circumvented by running each pipeline operation as an independent subprocess, communicating via shared memory and TCP sockets. This architecture allows for soft real-time monitoring of intermediate data products, efficient CPU usage, compatibility with custom GUIs, and the use of the Python interpreter as a simple RTC interface.

While extreme AO applications may still require custom C++ solutions, pyRTC is envisioned for a wider range of applications, including:

- Moderate performance adaptive optics applications (approximately 1 kHz speed for about 100 modes).
- Lab environments dependent on student labor.
- Test systems for hardware/software at on-sky speeds.
- Any neural network/AI-based controller built in Python.

## Features and Implementation

pyRTC is structured in order to minimize the amount of additional coding required to integrate pyRTC into a new hardware environment. The way pyRTC accomplishes this is by defining abstract superclasses for AO components, namely:

- Loop.py: Responsible for the AO integrator logic, relies on data products from the slopes process class.  
- ScienceCamera.py,  Responsible for the PSF logic, connects to the PSF camera and produces data products for further processing 
- SlopesProcess.py,  Responsible for the slopes computations, relies on data products from the WavefrontSensor class. 
- WavefrontCorrector.py,  Responsible for the controlling the Deformable Mirror, receives commands from Loop class or elsewhere.
- WavefrontSensor.py,  Responsible for the WFS logic, connects to the WFS camera and produces data products for the SlopesProcess class. 

These superclasses are then overridden by the user defined `hardware` class which interfaces with the hardware's API. We have provided examples in the `pyRTC/hardware` folder. Ideally, users will contribute their hardware examples and the repository will serve as a library of examples for new users to follow. For some of the components (e.g., SlopesProcess), users can choose to override the classes if they require specific computations or they can use the classes default functionality. We intend to expand the scope of the default functionality as new use cases emerge. 

Once the hardware classes have been established, a communication interface implemented in `Pipeline.py` allows the user to initialize their AO loop as either a set of independent processes which communicate via TCP (for performance, to get around the GIL), or within a single program (for simplicity). In either case, pyRTC has been written to be entirely initialized using a config YAML file. This includes the functions which will be included in the main RTC pipeline. Therefore, once the core hardware compatibility has been written, all of the real-time manipulation of the system is to be done via iPython interface, or via config file changes.

### Shared Memory and Live Viewing 

pyRTC is built using shared memory objects provided by the `multiprocessing` python package. Therefore, all data products shared between pyRTC components are available for soft real-time viewing and analysis. pyRTC comes with a real-time viewing script called `pyRTCView.py` which utilizes the pyQT5 package to produce a live feed of a specific shared memory object. For example, to view the images produced by the WavefrontSensor class run:

```
python pyRTCView.py wfs
```

We hope to expand this viewer into an example GUI in the future. 

---

# Acknowledgements

<!-- We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project. -->

# References

