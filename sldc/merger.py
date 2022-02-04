# -*- coding: utf-8 -*-
from functools import partial
import numpy as np
from collections import defaultdict
from shapely.geometry import JOIN_STYLE, box as bbox
from shapely.ops import unary_union
from shapely import affinity

from .util import shape_array

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__contributor__ = ["Romain Mormont <romainmormont@hotmail.com>"]
__version = "0.1"


class UnionFind(object):
    def __init__(self, elements):
        self._nodes = {e: (e, 0) for e in elements}
    
    def union(self, elem1, elem2):
        if not self.has(elem1) or not self.has(elem2):
            return False
        parent1, rank1 = self._find(elem1)
        parent2, rank2 = self._find(elem2)
        
        if parent1 == parent2:
            return True

        if rank1 > rank2: 
            self._nodes[parent1] = (parent2, rank1)
            self._nodes[parent2] = (parent2, rank2 + 1)
            return parent2
        else:
            self._nodes[parent2] = (parent1, rank2) 
            self._nodes[parent1] = (parent1, rank1 + 1)
            return parent1

    def same(self, elem1, elem2):
        parent1 = self.find(elem1)
        return parent1 is not None and parent1 == self.find(elem2)

    def has(self, elem):
        return elem in self._nodes

    def _find(self, elem):
        if not self.has(elem):
            return None, -1
        parent, rank = self._nodes[elem]
        if parent == elem:
            return parent, rank
        else:
            pparent, prank = self._find(parent)
            self._nodes[elem] = (pparent, rank)
            return pparent, prank

    def find(self, elem):
        parent, _ = self._find(elem)
        return parent

    def connected_components(self):
        comp_dict = defaultdict(list)
        for elem, (parent, _) in self._nodes.items():
            if elem == parent:
                comp_dict[elem].append(elem)
            else:
                comp_dict[self.find(elem)].append(elem)
        return comp_dict.values()


def aggr_max_area_label(areas, labels):
    unique_labels = np.unique(labels)
    max_area, max_label = -1, -1
    for l in unique_labels:
        area = np.sum(areas[labels == l])
        if area > max_area:
            max_area = area
            max_label = l
    return max_label


class TilePolygons(object):
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3
    NONE = 4
    
    def __init__(self, tile_id, topology, polygons_dict, filter_dist=None):
        super().__init__()
        self._tile_id = tile_id
        self._topology = topology
        self._polygons_dict = polygons_dict
        self._filter_dist = filter_dist
        self._by_side = self._compute_by_side()

    def _compute_by_side(self):
        """Group polygons based on whether they touch the side of the tile or not"""
        tile = self._topology.tile(self._tile_id)
        off_x, off_y = tile.offset
        translate = partial(affinity.translate, xoff=off_x, yoff=off_y)
        boxes = {
            self.TOP: translate(bbox(0, 0, tile.width, self._filter_dist)),
            self.BOTTOM: translate(bbox(0, tile.height - self._filter_dist, tile.width, tile.height)),
            self.LEFT: translate(bbox(0, 0, self._filter_dist, tile.height)),
            self.RIGHT: translate(bbox(tile.width - self._filter_dist, 0, tile.width, tile.height))
        }
        by_side = defaultdict(list)
        for poly_id, polygon in self._polygons_dict.items():
            matched_any = False
            for side, box in boxes.items():
                if box.intersects(polygon):
                    by_side[side].append(poly_id)
                    matched_any = True
            if not matched_any:
                by_side[self.NONE].append(poly_id)
        return by_side

    @property
    def polygons(self):
        return self._polygons

    def polygons_by_side(self, side):
        if self._filter_dist is None:
            # not filter dist -> return all poly
            return self.polygons 
        return self._by_side[side]

    @classmethod
    def opposite_side(cls, side):
        return {cls.TOP: cls.BOTTOM, cls.BOTTOM: cls.TOP, cls.LEFT: cls.RIGHT, cls.RIGHT: cls.LEFT}.get(side)


class SemanticMergingPolicy(object):
    """Merging policy for semantic merger. Specify the strategy to apply with regard to close polygons that
    have different labels. 
    """
    POLICY_NO_MERGE = "no_merge"  # TODO implement other policies


class SemanticMerger(object):
    """A class for merging labelled polygons. Close polygons having the same label are merged.
    """
    def __init__(self, tolerance, policy=SemanticMergingPolicy.POLICY_NO_MERGE):
        """Constructor for Merger objects

        Parameters:
        -----------
        tolerance: int
            Maximal distance between two polygons so that they are considered from the same object
        policy: str
            A merging policy to apply for overlapping polygons with different classes
        """
        self._tolerance = tolerance
        self._policy = policy

    def merge(self, tiles, polygons, tile_topology, labels=None):
        """Merge the polygons passed in a per-tile fashion according to the tile topology

        Parameters
        ----------
        tiles: iterable of tile identifiers (size: n, subtype: int)
            The identifiers of the tiles containing the polygons to merge
        polygons: iterable (size: n, subtype: iterable of shapely.geometry.Polygon)
            The polygons to merge provided as an iterable of iterables. The iterable i in polygons contains all 
            the polygons detected in the tile tiles[i].
        tile_topology: TileTopology
            The tile topology that was used to generate the tiles passed in polygons_tiles
        labels: iterable (size: n, subtype: iterable of int, default: None)
            The labels associated with the polygons. If None, all polygons are considered to have the same label.
            
        Returns
        -------
        polygons: iterable (size: m, subtype: shapely.geometry.Polygon)
            An iterable of polygons objects containing the merged polygons
        out_labels: iterable (size: m, subtype: int)
            The labels of the merged polygons. If labels was None, this return value is omitted.
        """
        tiles_dict, polygons_dict, labels_dict = self._build_dicts(tiles, polygons, tile_topology, labels=labels)
        # no polygons
        if len(polygons_dict) <= 0:
            return np.array([]) if labels is None else (np.array([]), np.array([]))
        
        # stores the polygons indexes as nodes
        geom_uf = UnionFind(polygons_dict.keys())
        
        # add edges between polygons that should be merged
        for tile_id, tile_polygons in tiles_dict.items():
            # check whether polygons in neighbour tiles must be merged 
            for side, neighbour in enumerate(tile_topology.tile_neighbours(tile_id)):
                if neighbour is None:
                    continue
                curr_candidates = tile_polygons.polygons_by_side(side)
                neigh_candidates = tiles_dict[neighbour].polygons_by_side(TilePolygons.opposite_side(side))
                self._register_merge(curr_candidates, neigh_candidates, polygons_dict, labels_dict, geom_uf)
        
        merged_polygons, merged_labels = self._do_merge(geom_uf, polygons_dict, labels_dict)
        if labels is None:
            return shape_array(merged_polygons)
        else:
            return shape_array(merged_polygons), np.array(merged_labels)

    def _register_merge(self, polygons1, polygons2, polygons_dict, labels_dict, geom_uf):
        """Compare 2-by-2 the polygons in the two arrays. If they are very close (using `self._tolerance` as distance
        threshold) and can be merged regarding their labels and the merging policy, they are registered as polygons to 
        be merged in the geometry graph (the registration being an edge between the nodes corresponding to the polygons 
        in geom_graph).

        Parameters
        ----------
        polygons1: iterable
            Iterable of integers containing polygons indexes
        polygons2: iterable
            Iterable of integers containing polygons indexes
        polygons_dict: dict
            Dictionary mapping polygon identifiers with actual shapely polygons objects
        labels_dict: dict
            Dictionnary mapping polygon ids with their labels
        geom_graph: UnionFind
            Disjoint set structure for registering meregs
        """
        for poly_id1 in polygons1:
            for poly_id2 in polygons2:
                if geom_uf.same(poly_id1, poly_id2):
                    continue
                poly1, poly2 = polygons_dict[poly_id1], polygons_dict[poly_id2]
                label1, label2 = labels_dict[poly_id1], labels_dict[poly_id2]
                if poly1.distance(poly2) < self._tolerance and label1 == label2:
                    geom_uf.union(poly_id1, poly_id2)

    def _do_merge(self, geom_uf, polygons_dict, labels_dict):
        """Effectively merges the polygons that were registered to be merged in the geom_graph Graph and return the
        resulting polygons in a list.

        Parameters
        ----------
        geom_uf: UnionFind
            A disjoint set structure with registered merges
        polygons_dict: dict
            Dictionary mapping polygon identifiers with actual shapely polygons objects
        labels_dict: dict
            Dictionnary mapping polygon ids with their labels

        Returns
        -------
        polygons: iterable
            An iterable of polygons objects containing the merged polygons
        """
        dilation_dist = self._tolerance
        join = JOIN_STYLE.mitre
        merged_polygons = []
        merged_labels = []
        for component in geom_uf.connected_components():
            if len(component) == 1:
                polygon = polygons_dict[component[0]]
                label = labels_dict[component[0]]
            else:
                polygons = [polygons_dict[poly_id].buffer(dilation_dist, join_style=join) for poly_id in component]
                polygon = unary_union(polygons).buffer(-dilation_dist, join_style=join)
                # determine label (take label representing the largest area)
                areas = np.array([polygons_dict[poly_id].area for poly_id in component])
                labels = np.array([labels_dict[poly_id] for poly_id in component])
                label = aggr_max_area_label(areas, labels)
            merged_polygons.append(polygon)
            merged_labels.append(label)
        return merged_polygons, merged_labels

    def _build_dicts(self, tiles, polygons, topology, labels=None):
        """Given a array of tuples (polygons, tile), return dictionaries for executing the merging:

        Parameters
        ----------        
        tiles: iterable of tile identifiers (size: n, subtype: int)
            The identifiers of the tiles containing the polygons to merge
        polygons: iterable (size: n, subtype: iterable of shapely.geometry.Polygon)
            The polygons to merge provided as an iterable of iterables. The iterable i in polygons contains all 
            the polygons detected in the tile tiles[i].
        labels: iterable (size: n, subtype: iterable of int, default: None)
            The labels associated with the polygons. If None, all polygons are considered to have the same label.
        topology: TileTopology
            The tile topology

        Returns
        -------
        polygons_dict: dictionary
            Maps a unique integer identifier with a polygon. All the polygons passed to the functions are given an
            identifier and are stored in this dictionary
        tiles_dict:
            Maps a tile identifier with the an array containing the ids of the polygons located in this tile.
        """
        tiles_dict = dict()
        polygons_dict = dict()
        labels_dict = dict()

        polygon_cnt = 1
        for i, (tile_id, polygons) in enumerate(zip(tiles, polygons)):

            curr_tile_poly_dict = dict()
            for j, polygon in enumerate(polygons):
                curr_tile_poly_dict[polygon_cnt] = polygon
                labels_dict[polygon_cnt] = 1 if labels is None else labels[i][j]
                polygon_cnt += 1

            tiles_dict[tile_id] = TilePolygons(tile_id, topology, curr_tile_poly_dict, filter_dist=self._tolerance + topology.overlap)
            polygons_dict.update(curr_tile_poly_dict)

        return tiles_dict, polygons_dict, labels_dict
