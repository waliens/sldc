import sys
from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()

__version__ = "1.2.1"

# fix versions for dependencies for python 2.7 (otherwise some tests fail)
if sys.version_info[0] == 2:
    requires = ['rasterio', 'affine', 'pillow == 5.1', 'numpy == 1.15', 'joblib == 0.11', 'geos', 'shapely == 1.6.4',
                'scipy == 1.1', 'scikit-image', 'opencv-python-headless']
else:  # python 3
    requires = ['rasterio', 'affine', 'pillow', 'numpy', 'joblib', 'geos', 'shapely', 'scipy', 'scikit-image',
                'opencv-python-headless']

setup(
    name='sldc',
    version=__version__,
    description='SLDC, a generic framework for object detection and classification in large images.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['sldc'],
    url='https://github.com/waliens/sldc',
    author="Romain Mormont",
    author_email="romain.mormont@gmail.com",
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    install_requires=requires
)
