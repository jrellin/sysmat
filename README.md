# System Response Generation Code for Slit/Pinhole Collimated Imaging System

Simple package to generate imaging system responses. 'imager_system.py' can be run from terminal with set parameters using main() as an example for any arbitrary configutation of detectors, slits/pinholes, and source geometries

Current Version: This branch is meant to be a cleaned up version of master with removal of superfluous code and fixes/speedups in system response calculations

## Current Requirements

1. Python 3
2. Numpy
3. Pytables

### Note
Default orientation assumes +z axis points from detector plane to source plane

Can generate large files depending on parameter space (> GB). Examples available on request
