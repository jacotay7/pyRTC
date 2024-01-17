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

---

<!-- # Citations -->

<!-- Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% } -->

# Acknowledgements

<!-- We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project. -->

# References
