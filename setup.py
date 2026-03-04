from pathlib import Path

from setuptools import find_packages, setup


README = Path(__file__).parent / "README.md"

setup(
    name='pyRTC',
    version='1.0.0',
    description='An object-oriented adaptive optics real-time control software written in Python.',
    long_description=README.read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    url='https://github.com/jacotay7/pyRTC',
    author='Jacob Taylor',
    author_email='jacob.taylor@mail.utoronto.ca',
    license='GPL-3.0-or-later',
    packages=find_packages(include=['pyRTC', 'pyRTC.*']),
    include_package_data=True,
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.26,<2.3',
        'matplotlib',
        'PyYAML',
        'numba',
        'scipy',
        'psutil',
        'optuna',
        'cmaes',
        'numexpr',
        'astropy',
    ],
    extras_require={
        "gpu": [
            "torch",
        ],
        "torch": [
            "torch",
        ],
        "viewer": [
            "PyQt5",
            "tqdm",
        ],
        "hardware": [
            "pipython",
            "rotpy",
            "ximea",
        ],
        "docs": [
                "sphinx",
                "sphinx-autobuild",
                "sphinx-rtd-theme",
                # For spelling
                "sphinxcontrib.spelling",
                # Copy button for code snippets
                "sphinx_copybutton",
            ],
    },
    entry_points={
        "console_scripts": [
            "pyrtc-view=pyRTC.scripts.view:main",
            "pyrtc-shm-monitor=pyRTC.scripts.shm_monitor:main",
            "pyrtc-clear-shms=pyRTC.scripts.clear_shms:main",
            "pyrtc-view-launch-all=pyRTC.scripts.view_launch_all:main",
            "pyrtc-measure-latency=pyRTC.scripts.measure_latency:main",
        ]
    },

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
)
