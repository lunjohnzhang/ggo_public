import os

os.environ["CONFIG_ROOT_PATH"] = "WPPL/configs/"

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
CONFIG_DIR = os.path.join(_parent_dir, "config")
LOG_DIR = os.path.join(_parent_dir, "logs")
MAP_DIR = os.path.join(_parent_dir, "maps")
COMPETITION_DIR = os.path.join(_parent_dir, "WPPL")
G_GRAPH_OUT_DIR= os.path.join(_parent_dir, "g_graph")