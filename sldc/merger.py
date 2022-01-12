# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
from shapely.geometry import JOIN_STYLE, box
from shapely.ops import unary_union

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__contributor__ = ["Romain Mormont <romainmormont@hotmail.com>"]
__version = "0.1"


class Graph(object):
    """A class for representing a graph
    """
    def __init__(self):
        self.nodes = []
        self.node2idx = {}
        self.edges = {}

    def add_node(self, value):
        """Add a node to the graph

        Parameters
        ----------
        value: int
            Node value
        Returns
        -------
        index: int
            Return the node index
        """
        self.nodes.append(value)
        self.node2idx[value] = len(self.nodes) - 1
        return len(self.nodes) - 1

    def add_edge(self, source, destination):
        """Add an edge to the graph

        Parameters
        ----------
        source: int
            Id of the source node
        destination: int
            Id of the destination node
        """
        ls = self.edges.get(source, [])
        if len(ls) == 0:
            self.edges[source] = ls
        ls.append(destination)

    def connex_components(self):
        """Find the connex components of the graph
        Returns
        -------
        components: iterable (subtype: iterable of int)
            An iterable containing connex components. A connex component is an iterable of node indexes
        """
        visited = [False]*len(self.nodes)
        components = []
        stack = []  # store index of reachable nodes
        for node in self.node2idx.keys():
            current_comp = []
            stack.append(node)
            while len(stack) > 0:
                current_node = stack.pop()
                curr_idx = self.node2idx[current_node]
                if visited[curr_idx]:
                    continue
                visited[curr_idx] = True
                current_comp.append(current_node)
                stack.extend(self.edges.get(current_node, []))
            if len(current_comp) > 0:
                components.append(current_comp)
        return components

    def __getitem__(self, node_index):
        return self.nodes[node_index]


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
    OTHER = 4
    
    def __init__(self, tile_id, topology, polygons, filter_dist=None):
        super().__init__()
        self._tile_id = tile_id
        self._topology = topology
        self._all = polygons
        self._filter_dist = filter_dist
        self._by_side = self._compute_by_side(polygons)

    @classmethod
    def compute_by_side(cls, tile_id, polygons, topology, dist=1):
        """Group polygons based on whether they touch the side of the tile or not"""
        tile = topology.tile(tile_id)
        boxes = {
            cls.TOP: box(0, 0, tile.width, dist),
            cls.BOTTOM: box(0, tile.height - dist, tile.width, tile.height),
            cls.LEFT: box(0, 0, dist, tile.height),
            cls.RIGHT: box(tile.width - dist, 0, tile.width, tile.height)
        }
        by_side = defaultdict(list)
        for polygon in polygons:
            matched_any = False
            for side, box in boxes.items():
                if box.intersects(polygon):
                    by_side[side].append(polygon)
                    matched_any = True
            if not matched_any:
                by_side[cls.OTHER].append(polygon)
        return by_side

    @property
    def polygons(self):
        return self._polygons

    def poly_by_side(self, side):
        if self._filter_dist is None:
            # not filter dist -> return all poly
            return self.polygons 
        return self._by_side[side]


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
        tiles_dict, polygons_dict, labels_dict = SemanticMerger._build_dicts(tiles, polygons, labels=labels)
        # no polygons
        if len(polygons_dict) <= 0:
            return np.array([]) if labels is None else (np.array([]), np.array([]))
        # stores the polygons indexes as nodes
        geom_graph = Graph()
        # add polygons
        for index in polygons_dict.keys():
            geom_graph.add_node(index)
        # add edges between polygons that should be merged
        for tile_identifier in tiles_dict.keys():
            # check whether polygons in neighbour tiles must be merged
            neighbour_tiles = tile_topology.tile_neighbours(tile_identifier)
            for neighbour in neighbour_tiles:
                if neighbour is not None:
                    self._register_merge(tiles_dict[tile_identifier], tiles_dict[neighbour], polygons_dict, labels_dict, geom_graph)
        merged_polygons, merged_labels = self._do_merge(geom_graph, polygons_dict, labels_dict)
        if labels is None:
            return np.array(merged_polygons)
        else:
            return np.array(merged_polygons), np.array(merged_labels)

    def _register_merge(self, polygons1, polygons2, polygons_dict, labels_dict, geom_graph):
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
        geom_graph: Graph
            The graph in which must be registered the polygons to be merged
        """
        for poly_id1 in polygons1:
            for poly_id2 in polygons2:
                poly1, poly2 = polygons_dict[poly_id1], polygons_dict[poly_id2]
                label1, label2 = labels_dict[poly_id1], labels_dict[poly_id2]
                if poly1.distance(poly2) < self._tolerance and label1 == label2:
                    geom_graph.add_edge(poly_id1, poly_id2)

    def _do_merge(self, geom_graph, polygons_dict, labels_dict):
        """Effectively merges the polygons that were registered to be merged in the geom_graph Graph and return the
        resulting polygons in a list.

        Parameters
        ----------
        geom_graph: Graph
            The graph in which were registered the polygons to be merged
        polygons_dict: dict
            Dictionary mapping polygon identifiers with actual shapely polygons objects
        labels_dict: dict
            Dictionnary mapping polygon ids with their labels

        Returns
        -------
        polygons: iterable
            An iterable of polygons objects containing the merged polygons
        """
        components = geom_graph.connex_components()
        dilation_dist = self._tolerance
        join = JOIN_STYLE.mitre
        merged_polygons = []
        merged_labels = []
        for component in components:
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

    @classmethod
    def _build_dicts(cls, tiles, polygons, labels=None):
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
            polygons_ids = []

            for j, polygon in enumerate(polygons):
                polygons_dict[polygon_cnt] = polygon
                labels_dict[polygon_cnt] = 1 if labels is None else labels[i][j]
                polygons_ids.append(polygon_cnt)
                polygon_cnt += 1

            tiles_dict[tile_id] = polygons_ids

        return tiles_dict, polygons_dict, labels_dict
