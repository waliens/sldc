# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np

__author__ = "Romain Mormont <romainmormont@hotmail.com>"
__version = "0.1"


class SemanticSegmenter(object):
    """Interface to be implemented by any class which implement a segmentation algorithm.
    """
    __metaclass__ = ABCMeta

    def __init__(self, classes=None):
        """Constructor
        
        Parameters
        ----------
        classes: iterable (subtype: int)
            Classes produced by the segmenter.
        """
        self._classes = classes

    @abstractmethod
    def segment(self, image):
        """Segment the image using a custom segmentation algorithm
        Parameters
        ----------
        image: ndarray (shape: [width, height{, channels}])
            An NumPy representation of the image to segment.

        Returns
        -------
        segmented : ndarray (shape: [width, height], dtype=np.int32)
            An NumPy representation of the segmented image. Each pixel is associated with its class index.
            The class index is a number in {0, 1, ..., n_classes - 1} where n_classes is the number of classes produced 
            by the segmenter. The class index corresponds to the index of the class in `self.classes` if defined.
        """
        pass

    def segment_batch(self, images):
        """Segment a batch of images using a custom segmentation algorithm. Default implementation calls
        segment() iteratively on each individual images. Re-implement this method for actual batch segmentation.

        Parameters
        ----------
        images: ndarray (shape: [batch_size, width, height{, channels}])
            An NumPy representation of the images to segment.

        Returns
        -------
        segmented : ndarray (shape: [batch_size, width, height], dtype=np.int32)
            An NumPy representation of the segmented images. See `segment` for specification of the segmentation mask.
        """
        masks = [self.segment(image) for image in images]
        return np.array(masks)

    @property
    def n_classes(self):
        """The maximum number of classes this segmenter might produce in the segmentation map.
        Returns
        -------
        n_classes: int
            The (maximum) number of classes produced by this segmenter. -1 if undefined.
        """
        return len(self._classes) if self._classes is not None else -1

    @property
    def classes(self):
        """Return the classes produced by the segmenter (for a given image, the segmenter might output a subset
        of those classes)
        
        Returns
        -------
        classes: iterable (subtype: int)
            The classes expected to be produced by the segmenter.
        """
        return self._classes

    def _get_labels(self, mask):
        """Transform a mask of class index into a mask containing actual classification labels.
        Parameters
        ----------
        mask: ndarray (shape: [width, height])
            An NumPy representation of a segmentation mask. Each pixel should be a class index (see 
            `SemanticSegmenter.segment` function docstring). 
        
        Returns
        -------
        mask: ndarray (shape: [width, height])
            A NumPy representation of the mask containing the true labels of the image
        
        Raises
        ------
        ValueError: if the true labels were not defined
        """
        if self.classes is None:
            raise ValueError("Class labels are not defined.")
        return np.take(self.classes, mask)

    def true_segment(self, image):
        """Segment the image and produce a segmentation label containing the true classification labels
        
        Parameters
        ----------
        image: ndarray (shape: [width, height{, channels}])
            An NumPy representation of the image to segment.

        Returns
        -------
        segmented : ndarray (shape: [width, height], dtype=np.int32)
            An NumPy representation of the segmented image. Each pixel is associated with its true classification 
            label.
            
        Raises
        ------
        ValueError: if the true labels were not defined
        """
        return self._get_labels(self.segment(image))


class ProbabilisticSegmenter(SemanticSegmenter):
    """Interface to be implemented by any class which implements a probabilistic segmentation algorithm. 
    Such algorithm produces class probabilities for each pixel. 
    """
    def __init__(self, classes=None):
        super(ProbabilisticSegmenter, self).__init__(classes=classes)

    @abstractmethod
    def segment_proba(self, image):
        """Generate a segmentation mask containing pixel class probabilities
        
        Parameters
        ----------
        image: ndarray (shape: [width, height{, channels}])
            An NumPy representation of the image to segment.
        
        Returns
        -------
        probas: ndarray (shape: [width, height, `self.n_classes`])
            A probability map for each pixel of the image. The last channel is ordered as `self.classes`.
            If `self.n_classes` and `self.classes` are not defined, the last dimensions will be as large as 
            the number of classes predicted by the implemented algorithm
        """
        pass

    def segment(self, image):
        probas = self.segment_proba(image)
        return np.argmax(probas, axis=-1).astype(np.int32)


class Segmenter(SemanticSegmenter):
    """For backward-compatibility 
    
    Interface to be implemented by any class which implements a (binary) segmentation algorithm.
    
    Background pixels are represented by the value 0 ('black') while foreground ones are represented 
    by the value 255 ('white') by default.
    """
    __metaclass__ = ABCMeta

    def __init__(self, classes=None):
        super(Segmenter, self).__init__(classes=[0, 255] if classes is None else classes)
