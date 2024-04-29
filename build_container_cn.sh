# If you are in mainland China, you may use this script to build the container.
# Internally it uses the pip/apt package mirrors in mainland China to speed up
# the downloads.

sudo singularity build --sandbox singularity/ubuntu_warehouse/ singularity/ubuntu_warehouse_cn.def
sudo singularity run --writable singularity/ubuntu_warehouse
sudo singularity build singularity/ubuntu_warehouse.sif singularity/ubuntu_warehouse