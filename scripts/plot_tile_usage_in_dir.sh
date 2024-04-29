#!/bin/bash

TO_PLOT="$1"
DOMAIN="$2"

for DIR in ${TO_PLOT}/*;
do
    echo "${DIR}"
    bash scripts/plot_tile_usage.sh "$DIR" qd extreme $DOMAIN -p /media/project0
done