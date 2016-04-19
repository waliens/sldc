# -*- coding: utf-8 -*-

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


class ChainInformation(object):
    """ Stores information gathered at various stages of execution of the workflow.
    """
    def __init__(self):
        self._runs = dict()
        self._next_id = 1
        self._image2ids = dict()

    def __len__(self):
        """Return the number of workflow info registered till then
        Returns
        -------
        count: int
            The number of workflow info registered
        """
        return len(self._runs)

    def get_workflow_info(self, id):
        """Return the workflow info having a given id
        Parameters
        ----------
        id: int
            The id of the workflow info object to get

        Returns
        -------
        run: WorkflowInformation
            The workflow info object
        """
        return self._runs[id]

    def register_workflow_info(self, workflow_info, image):
        """Register a new run to the execution information object
        Parameters
        ----------
        workflow_info: WorkflowInformation
            The workflow information object
        image: int
            A unique identifier for the image on which the workflow was executed

        Returns
        -------
        result: WorkflowInformation
            The workflow information object with its id set
        """
        workflow_info.id = self._next_id
        self._image2ids[workflow_info.id] = self._image2ids.get(image, []) + [workflow_info.id]
        self._runs[workflow_info.id] = workflow_info
        self._next_id += 1  # increment the next run id
        return workflow_info

    def register_workflow_collection(self, collection, image):
        """Register a new run to the execution information object
        Parameters
        ----------
        collection: WorkflowInformationCollection
            The collection of workflow information objects
        image: int
            A unique identifier for the image on which the workflow was executed
        """
        for workflow_info in collection:
            self.register_workflow_info(workflow_info, image)

    def get_workflow_infos_by_image(self, image):
        """Return all the workflow information objects registered for the given image
        Parameters
        ----------
        image: int
            The unique identifier of the image (used when the workflows were registered)

        Returns
        -------
        list: list of WorkflowInformation
            The workflow information objects registered for the given image
        """
        return WorkflowInformationCollection([self.get_workflow_info(id) for id in self._image2ids.get(image, [])])


class WorkflowInformation(object):
    """Workflow information : execution about a workflow run. A run is a complete execution
    (segment, locate, dispatch and classify) of a single workflow and comprises the following information :
        - id : runs are assigned unique ids as they are notified to the ChainInformation object)
        - polygons : the polygons generated by a given run
        - dispatch : list of which the ith element matches the index of the dispatching rule that matched
            the ith polygon, -1 if none did
        - class : list of which the ith integer is the class predicted by the classifier to
            which was dispatched the ith polygon, -1 if none did
        - timing : the information about the execution time of the workflow
        - metadata : a comment from the implementor of the workflow to document how the previous data were generated
    """
    def __init__(self, polygons, dispatch, classes, timing, id=None, metadata=""):
        """Construct a run object
        Parameters
        ----------
        polygons: list
            The polygons generated by the run
        dispatch: list
            Their dispatch indexes
        classes: list
            Their predicted classes
        timing: SLDCTiming
            Execution time information
        id: int, (optional, default: None)
            The run id, None if to assign it later
        metadata: string, (optional, default: "")
            String data/comment to associate with the workflow generated data
        """
        self._id = id
        self._polygons = polygons
        self._dispatch = dispatch
        self._classes = classes
        self._metadata = metadata
        self._timing = timing

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def polygons(self):
        return self._polygons

    @property
    def dispatch(self):
        return self._dispatch

    @property
    def classes(self):
        return self._classes

    @property
    def metadata(self):
        return self._metadata

    @property
    def timing(self):
        return self._timing


class WorkflowInformationCollection(object):
    """An collection for storing workflow information objects
    """
    def __init__(self, items=None):
        """
        Parameters
        ----------
        items: list of WorlflowInformation
            Object to insert in the list, if not provided the collection is initialized empty
        """
        self._items = items if items is not None else []

    def __len__(self):
        return len(self._items)

    def __getitem__(self, item):
        return self._items[item]

    def __setitem__(self, key, value):
        self._items[key] = value

    def __iter__(self):
        for item in self._items:
            yield item

    def append(self, value):
        """Append the workflow information at the end of the collection
        Parameters
        ----------
        value: WorkflowInformation
            The object to append
        """
        self._items.append(value)

    def polygons_iterator(self):
        """An iterator that goes through all the polygons stored in the collection
        The yielded value is a tuple containing the polygon, the dispatch index and the predicted
        class
        """
        for item in self._items:
            for polygon, dispatch, cls in zip(item.polygons, item.dispach, item.classes):
                yield polygon, dispatch, cls
