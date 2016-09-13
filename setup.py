from distutils.core import setup

"""
Dependencies:
* open-cv2
* numpy
* Shapely
"""
setup(
    name="sldc",
    version="1.0",
    description="SLDC, a generic workflow for object detection and classification in large images.",
    author="Romain Mormont",
    author_email="romain.mormont@gmail.com",
    packages=["sldc"]
)
