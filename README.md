# SLDC

**_SLDC_** is a framework created for accelerating development of large image analysis workflows. It is especially well 
suited for solving more or less complex problems of object detection and classification in multi-gigapixel images.

The framework encapsulates problem-independent logic such as parallelism, memory constraints (due to large image handling) 
while providing a concise way of declaring problem-dependent components of the implementer's workflows.

[![Build status](https://travis-ci.org/waliens/sldc.svg?branch=master)](https://travis-ci.org/waliens/sldc)
[![codecov](https://codecov.io/gh/waliens/sldc/branch/master/graph/badge.svg)](https://codecov.io/gh/waliens/sldc)

## Documentation

The algorithm used by the framework as well as some toy examples are presented in the [Wiki](https://github.com/waliens/sldc/wiki).

## Dependencies

The framework currently works under Python 2.7 only, but we're working on extending the portability to Python 3.x.

The required dependencies are the following :

* Numpy (>= 1.10, might work with earlier versions)
* OpenCV (>= 3.0)
* Pillow (>= 3.1.1)
* joblib (>= 0.9.4)
* Shapely (>= 1.5.13)
* Scipy (>= 0.18.1)

## Install

### With anaconda environment

1) Install Anaconda/Miniconda: https://docs.continuum.io/anaconda/install

2) Set up the environment

 + For UNIX-based systems:
```bash
# Create envrionment and install packages
conda create -n sldc python=2.7 pillow numpy joblib shapely opencv scipy
# Activate environment
source activate sldc
```

 + For windows:
```bash
# Create environment and install packages
conda create -n sldc python=2.7 pillow numpy joblib scipy
# Activate environment
activate sldc
# Install shapely and opencv3
conda install -c conda-forge shapely
conda install -c menpo opencv3=3.1.0
```

3) Install sldc

 + Download the sources
  + Either by cloning the repository: `git clone https://github.com/waliens/sldc.git`
  + Or by downloading an archive: `https://github.com/waliens/sldc/archive/master.zip`
 + Move to the _SLDC_ sources root folder
 + Install sldc: `python setup.py install`

4) Check your install by running `python -c "import sldc"`

## Bindings

The library is image format agnostic and therefore allows you to integrate it with any existing image format by implementing some interfaces. However, some bindings were implemented for integrating SLDC with: 

+ [Cytomine](http://www.cytomine.be/): [`cytomine-sldc` repository](https://github.com/cytomine/Cytomine-python-datamining/tree/master/cytomine-datamining/algorithms/sldc) 
+ [OpenSlide](http://openslide.org/): [`sldc-openslide` repository](https://github.com/waliens/sldc-openslide)

## References

If you use _SLDC_ in a scientific publication, we would appreciate citations: [Mormont & al., Benelearn, 2016](http://orbi.ulg.ac.be/handle/2268/202624).

The framework was initially developed in the context of [this master thesis](http://hdl.handle.net/2268.2/1314).
