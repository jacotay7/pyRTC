from setuptools import setup

setup(
    name='pyRTC',
    version='1.0.0',    
    description='An object-oriented adaptive optics real-time control software written in Python. The goal is to be a universal and simple to use python package while maintaining enough real-time performance to be widely applicable within the AO community.',
    url='https://github.com/jacotay7/pyRTC',
    author='Jacob Taylor',
    author_email='jacob.taylor@mail.utoronto.ca',
    license='GNU',
    packages=['pyRTC','pyRTC.hardware'],
    install_requires=['numpy',
                      'matplotlib',
                      'PyYaml',
                      'numba',
                      'scipy',
                      'pyqt5'             
                      ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Environment :: MacOS X',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
