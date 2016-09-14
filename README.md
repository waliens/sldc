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