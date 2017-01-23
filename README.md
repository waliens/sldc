# SLDC

**_SLDC_** is a framework created for accelerating development of large image analysis workflows. It is especially well 
suited for solving more or less complex problems of object detection and classification in multi-gigapixel images.

The framework encapsulates problem-independent logic such as parallelism, memory constraints (due to large image handling) 
while providing a concise way of declaring problem-dependent components of the implementer's workflows.

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

## Install

### With anaconda environment

1) Install Anaconda/Miniconda: https://docs.continuum.io/anaconda/install

2) Set up the environment

 + For UNIX-based systems:
```bash
# Create envrionment and install packages
conda create -n sldc python=2.7 pillow numpy joblib shapely
# Activate environment
source activate sldc
# Install opencv 3
conda install -c menpo opencv3=3.1.0
```

 + For windows:
```bash
# Create environment and install packages
conda create -n sldc python=2.7 pillow numpy joblib
# Activate environment
activate sldc
# Install shapely and opencv3
pip install -i https://pypi.anaconda.org/pypi/simple shapely
conda install -c menpo opencv3=3.1.0
```

3) Install sldc

 + Download the sources
 + Move to the _SLDC_ sources root folder
 + Install sldc: `python setup.py install`

4) Check your install by running `python -c "import sldc"`


## References

The framework was developed in the context of this master thesis: http://hdl.handle.net/2268.2/1314.
