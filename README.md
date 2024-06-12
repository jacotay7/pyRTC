# What is pyRTC?

Adaptive optics (AO) is a technology that rapidly detects and corrects optical aberrations to significantly improve the resolution of an optical system. AO has been applied to imaging problems in fields such as astronomy, ophthalmology, and microscopy, as well as various military and industrial applications. AO systems operate using a so-called `real-time controller' (RTC), a term used by the community to describe software responsible for converting optical aberration measurements into corresponding corrections. 

pyRTC is an open-source, community-driven Python package for real-time control of AO systems, built with the following core goals:

- **Customizable High-Performance AO Pipeline:** Provide an efficient RTC pipeline with potential for full user customization.
- **Abstraction of Core AO System Components:** Facilitate support for a broad range of AO system architectures.
- **Open Library of API Examples:** Provide a library of examples for common hardware APIs used by the community to save time implementing basic hardware interactions.
- **Real-Time Monitoring and Interface Flexibility:** Support real-time access to intermediate data products, text-based user interaction, and straightforward integration with user-built GUIs.
- **Cross-Platform Compatibility:** Ensure broad usability across different operating systems.

# Why should you use pyRTC?

pyRTC was built as a middle ground between performance and usability. Therefore, pyRTC can is intended for a wider range of applications, including:

- Moderate performance adaptive optics applications (approximately 1 kHz speed for a few hundred modes).
- AO Lab environments
- Test systems for hardware/software at on-sky speeds.
- Any neural network/AI-based controller built in Python.

# Installation

To install pyRTC, start by cloning the repository: 

```
git clone https://github.com/jacotay7/pyRTC.git
```

Navigate to the folder, and install with pip.

```
cd pyRTC
pip install .
```

Optionally, install with docs libraries:

```
cd pyRTC
pip install .[docs]
```

# Getting Started 


## Documentation

You can build the docs locally by following these steps:

First, install the required packages, this can be done by:

```
cd pyRTC
pip install .[docs]
```

See Installation Instructions for more info.

Next, navigate to the pyRTC folder and build the docs:

```
cd ~/pyRTC/docs/source
make html
```

The docs will be built in the `docs/source/_build/html` folder. You can run them locally with:

```
sphinx-autobuild . _build/html
```

This process can also be done using the including `build_and_run.sh` script located in the `docs/source` folder.

```
cd ~/pyRTC/docs/source
./build_and_run.sh
```

## Hardware

Since each AO system utilizes a unique hardware configuration, every new AO system will require a programmer to write interfacing code to control the hardware. This is unavoidable. Therefore, one of pyRTC's goals is compile a fairly complete set of API implementations for common hardware components like ALPAO DM's, PI modulators, and FLIR cameras so that community members will have easy access to existing exemplars. However, there is a significantly prohibitive cost associated with buying one of each possible hardware component an AO research might need to interact with. Therefore, we will rely on the community to provide their pyRTC implementations for their hardware components so that the whole community can benefit.

`pyRTC/hardware` hosts examples of specific hardware implementations we have created to date, which hopefully more to be added over time.

## Simluating Hardware

In order to test pyRTC without access to the specific AO hardware required, we have provided some examples of how to run pyRTC with a simulated AO system.

For our examples we will be using the open-source AO simulation software OOPAO developed by Héritier, C. et al. (https://github.com/cheritier/OOPAO/tree/master). Please refer to their repository for installation instructions. 

## Examples

We have provided an example of how to set up a single conjugate AO (SCAO) system using a Pyramid Wavefront Sensor. This example uses the OOPAO simulation software to simulate the hardware components of an AO system so that pyRTC can be tested without the need of real AO hardware. All examples can be found under the `examples` folder.

# Contributing

pyRTC is an open-source software intended to be built for and by the AO community. Please use pyRTC freely and create new branches for your systems. We ask that if you implement new features that may be of broader interest to the community while integrating pyRTC into your AO system, please contribute to the code base by opening up an issue or pull request on github.   

# Contact

For feedback and feature requests you can contact me via e-mail at jacob.taylor@mail.utoronto.ca.

## Citation

Please cite us if you use this code for your paper:

```
add paper info
```
