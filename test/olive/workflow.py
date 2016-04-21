from sldc.chaining import ImageProvider, PostProcessor, WorkflowExecutor
from pizza_workflow import ObjectType


class PizzaImageProvider(ImageProvider):
    def __init__(self, images):
        ImageProvider.__init__(self)
        self._images = images

    def get_images(self):
        return self._images


class PizzaOliveWorkflowExecutor(WorkflowExecutor):
    def after(self, sub_image, workflow_information):
        return

    def get_images(self, image, workflow_information):
        images = []
        for polygon, dispatch, cls in workflow_information.polygons_iterator():
            if cls == ObjectType.BIG_RED or cls == ObjectType.BIG_YELLOW:
                minx, miny, maxx, maxy = polygon.bounds
                images.append(image.window((int(miny), int(minx)), int(maxx - minx), int(maxy - miny)))
        return images


class PizzaPostProcessor(PostProcessor):
    def __init__(self, results):
        self._results = results

    @property
    def results(self):
        return self._results

    def post_process(self, image, workflow_information):
        self._results.extend(workflow_information)
