import os
import json
import numpy as np


class Map:
    """Data structure used for edge weight format convertion."""

    def __init__(self, fp):
        self.fp = fp
        self.fn = os.path.split(self.fp)[-1]
        self.height = None
        self.width = None
        self.graph = None

        self.load(self.fp)

    def load(self, fp: str):
        self.fp = fp
        if fp.endswith(".map"):
            with open(fp, "r") as f:
                # skip the type line
                f.readline()
                self.height = int(f.readline().split()[-1])
                self.width = int(f.readline().split()[-1])
                self.graph = np.zeros((self.height, self.width), dtype=np.int)
                # skip the map line
                f.readline()
                for row in range(self.height):
                    line = f.readline().strip()
                    assert len(line) == self.width
                    for col, loc in enumerate(line):
                        # obstacle
                        if loc == "@" or loc == "T":
                            self.graph[row, col] = 1
        elif fp.endswith(".json"):
            with open(fp, "r") as f:
                map_json = json.load(f)
                self.height = map_json["n_row"]
                self.width = map_json["n_col"]
                self.graph = np.zeros((self.height, self.width), dtype=np.int)
                for row in range(self.height):
                    line = map_json["layout"][row]
                    assert len(line) == self.width
                    for col, loc in enumerate(line):
                        # obstacle
                        if loc == "@" or loc == "T":
                            self.graph[row, col] = 1
        # self.print_graph(self.graph)

    def print_graph(self, graph: np.ndarray):
        map = ""
        height, width = graph.shape
        for i in range(height):
            for j in range(width):
                map += str(graph[i, j])
            map += "\n"
        print(map)


def compress_weights_with_wait_costs(map, full_weights):
    compressed_wait_costs = []
    for y in range(map.height):
        for x in range(map.width):
            weight_idx = y * map.width + x

            if map.graph[y, x] == 1:
                continue

            compressed_wait_costs.append(full_weights[5 * weight_idx + 4])

    # print("wait costs size", compressed_wait_costs.__len__())

    # compressed_wait_costs = "[" + ",".join(
    #     [str(w) for w in compressed_wait_costs]) + "]"
    compressed_wait_costs = json.dumps(compressed_wait_costs)

    compressed_weights = []

    for y in range(map.height):
        for x in range(map.width):
            weight_idx = y * map.width + x

            if map.graph[y, x] == 1:
                continue

            if (x + 1) < map.width and map.graph[y, x + 1] == 0:
                compressed_weights.append(full_weights[5 * weight_idx + 0])

            if (y - 1) >= 0 and map.graph[y - 1, x] == 0:
                compressed_weights.append(full_weights[5 * weight_idx + 3])

            if (x - 1) >= 0 and map.graph[y, x - 1] == 0:
                compressed_weights.append(full_weights[5 * weight_idx + 2])

            if (y + 1) < map.height and map.graph[y + 1, x] == 0:
                compressed_weights.append(full_weights[5 * weight_idx + 1])

            # if (weight_idx == 0):
            #     print(compressed_weights)

    # print("edge weights size", compressed_weights.__len__())

    # compressed_weights = "[" + ",".join([str(w)
    #                                      for w in compressed_weights]) + "]"

    compressed_weights = json.dumps(compressed_weights)
    return compressed_wait_costs, compressed_weights


def comp_compress_vertex_matrix(map, vertex_matrix):
    assert map.width * map.height == len(vertex_matrix)
    compressed_vertex_matrix = []
    for y in range(map.height):
        for x in range(map.width):
            pos = y * map.width + x
            if map.graph[y, x] == 1:
                continue
            compressed_vertex_matrix.append(vertex_matrix[pos])
    return compressed_vertex_matrix


def comp_compress_edge_matrix(map, edge_matrix):
    # edge matrix: h*w*[right,up,left,down]
    assert map.width * map.height * 4 == len(edge_matrix)

    compressed_edge_matrix = []
    for y in range(map.height):
        for x in range(map.width):
            pos = y * map.width + x
            if map.graph[y, x] == 1:
                continue

            if (x + 1) < map.width and map.graph[y, x + 1] == 0:  # right
                compressed_edge_matrix.append(edge_matrix[4 * pos + 0])

            if (y - 1) >= 0 and map.graph[y - 1, x] == 0:  # up
                compressed_edge_matrix.append(edge_matrix[4 * pos + 1])

            if (x - 1) >= 0 and map.graph[y, x - 1] == 0:  # left
                compressed_edge_matrix.append(edge_matrix[4 * pos + 2])

            if (y + 1) < map.height and map.graph[y + 1, x] == 0:  # down
                compressed_edge_matrix.append(edge_matrix[4 * pos + 3])
    return compressed_edge_matrix


def comp_uncompress_vertex_matrix(map, compressed_vertex_matrix, fill_value=0):
    j = 0
    vertex_matrix = []
    for y in range(map.height):
        for x in range(map.width):
            if map.graph[y, x] == 1:
                vertex_matrix.append(fill_value)
            else:
                vertex_matrix.append(compressed_vertex_matrix[j])
                j += 1
    return vertex_matrix


def comp_uncompress_edge_matrix(map, compressed_edge_matrix, fill_value=0):
    # edge matrix: h*w*[right,up,left,down]

    j = 0
    edge_matrix = []
    for y in range(map.height):
        for x in range(map.width):
            if map.graph[y, x] == 1:
                for i in range(4):
                    edge_matrix.append(fill_value)
            else:
                if (x + 1) < map.width and map.graph[y, x + 1] == 0:  # right
                    edge_matrix.append(compressed_edge_matrix[j])
                    j += 1
                else:
                    edge_matrix.append(fill_value)

                if (y - 1) >= 0 and map.graph[y - 1, x] == 0:  # up
                    edge_matrix.append(compressed_edge_matrix[j])
                    j += 1
                else:
                    edge_matrix.append(fill_value)

                if (x - 1) >= 0 and map.graph[y, x - 1] == 0:  # left
                    edge_matrix.append(compressed_edge_matrix[j])
                    j += 1
                else:
                    edge_matrix.append(fill_value)

                if (y + 1) < map.height and map.graph[y + 1, x] == 0:  # down
                    edge_matrix.append(compressed_edge_matrix[j])
                    j += 1
                else:
                    edge_matrix.append(fill_value)

    return edge_matrix
