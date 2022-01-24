
import numpy as np
from timeit import default_timer as timer
from skimage.io import imsave
from shapely.geometry import Polygon
from PIL.Image import fromarray
from PIL.ImageDraw import ImageDraw
from sldc import Image, Segmenter
from sldc.builder import SSLWorkflowBuilder


class NumpyImage(Image):
    def __init__(self, np_image):
        """An image represented as a numpy ndarray"""
        self._np_image = np_image

    @property
    def np_image(self):
        return self._np_image

    @property
    def channels(self):
        shape = self._np_image.shape
        return shape[2] if len(shape) == 3 else 1

    @property
    def width(self):
        return self._np_image.shape[1]

    @property
    def height(self):
        return self._np_image.shape[0]



def draw_poly(image, polygon, color=255):
  """Draw a polygon in the given color at the given location"""
  pil_image = fromarray(image)
  validated_color = color
  draw = ImageDraw(pil_image)
  if len(image.shape) > 2 and image.shape[2] > 1:
      validated_color = tuple(color)
  draw.polygon(polygon.boundary.coords, fill=validated_color, outline=validated_color)
  return np.asarray(pil_image)


def draw_square_by_corner(image, side, top_left, color):
  top_left = (top_left[1], top_left[0])
  top_right = (top_left[0] + side, top_left[1])
  bottom_left = (top_left[0], top_left[1] + side)
  bottom_right = (top_left[0] + side, top_left[1] + side)
  p = Polygon([top_left, top_right, bottom_right, bottom_left, top_left])
  return draw_poly(image, p, color)


def draw_all_poly(image, dim=10, vgap=2):
  h, w = image.shape
  i = 0
  shifted = False
  print("drawing", end="", flush=True)
  pcount = 0
  while i < h - dim:
    j = 0
    while j < w - dim:
      image = draw_square_by_corner(image, dim, (i, j + (dim if shifted else 0)), color=255)
      pcount += 1
      j += 2 * dim
    i += dim + vgap
    shifted = shifted ^ True
    print("\rdrawing: {:3.2f}%".format(100 * i / h), end="", flush=True)
  print()
  print(pcount)
  return image, pcount


class MySegmenter(Segmenter):
  def segment(self, mask):
    return (mask > 0).astype(np.uint8)  



def benchmark(img, tile_div=10):
  image = NumpyImage(img)

  builder = SSLWorkflowBuilder()
  builder.set_distance_tolerance(1)
  builder.set_overlap(0)
  builder.set_tile_size(img.shape[0] // tile_div, img.shape[1] // tile_div)
  builder.set_background_class(0)
  builder.set_n_jobs(1)
  builder.set_segmenter(MySegmenter())
  workflow = builder.get()

  times = list()
  n_tests = 10

  for _ in range(n_tests):
    start = timer()
    results = workflow.process(image)
    times.append(timer() - start)

  print("processed in {}s".format(sum(times) / n_tests))


def many_small():
  h, w = 2000, 2000
  np_image = np.zeros([h, w], dtype=np.uint8)
  np_image, _ = draw_all_poly(np_image)

  print("-------------------------------------")
  print("img 500x500, tiles 100x100")
  benchmark(np_image[:500, :500], tile_div=5)
  print("--")
  print("img 1000x1000, tiles 100x100")
  benchmark(np_image[:1000, :1000], tile_div=10)
  print("--")
  print("img 2000x2000, tiles 100x100")
  benchmark(np_image, tile_div=20)
  print("--")
  print("img 2000x2000, tiles 200x200")
  benchmark(np_image, tile_div=10)
  print("--")

def few_large():
  h, w = 2000, 2000
  np_image = np.zeros([h, w], dtype=np.uint8)
  np_image, _ = draw_all_poly(np_image, dim=h // 20)

  print("-------------------------------------")
  print("img 500x500, tiles 100x100")
  benchmark(np_image[:500, :500], tile_div=5)
  print("--")
  print("img 1000x1000, tiles 100x100")
  benchmark(np_image[:1000, :1000], tile_div=10)
  print("--")
  print("img 2000x2000, tiles 100x100")
  benchmark(np_image, tile_div=20)
  print("--")
  print("img 2000x2000, tiles 200x200")
  benchmark(np_image, tile_div=10)
  print("--")


if __name__ == "__main__":
  few_large()
  many_small()
  
  

"""
   | SLDC | Poly |   1.3   |   1.4   |
---|------|------|---------|---------|
 M |  (a) |  ~1k |  1.324s |  0.254s |
 A |  (b) |  ~4k |  5.576s |  1.062s |
 N |  (c) | ~17k | 25.430s |  4.494s |
 Y |  (d) | ~17k | 59.562s |  3.110s |
---|------|------|---------|---------|
 F |  (a) |  ~12 |  0.051s |  0.055s |
 E |  (b) |  ~50 |  0.182s |  0.223s |
 W |  (c) | ~200 |  0.791s |  1.001s |
   |  (d) | ~200 |  0.371s |  0.402s |

(a)   500 x 500 pxls, 100 x 100 tile size
(b) 1000 x 1000 pxls, 100 x 100 tile size
(d) 2000 x 2000 pxls, 100 x 100 tile size
(c) 2000 x 2000 pxls, 200 x 200 tile size
"""