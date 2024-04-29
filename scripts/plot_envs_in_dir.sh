#!/bin/bash

TO_PLOT="$1"
DOMAIN="$2"

shift 2
while getopts "p:" flag; do
  case "$flag" in
      p) PROJECT_DIR=$OPTARG;;
      *) echo "Invalid option. ${USAGE}"
  esac
done

mkdir -p "${TO_PLOT}/img"

SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"
if [ -n "$PROJECT_DIR" ]; then
  SINGULARITY_OPTS="$SINGULARITY_OPTS --bind ${PROJECT_DIR}:${PROJECT_DIR}"
fi
for ENV in ${TO_PLOT}/*.json;
do
    singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
        python env_search/analysis/visualize_env.py \
            --map-filepath "${ENV}" \
            --store_dir "${TO_PLOT}/img" \
            --domain "${DOMAIN}"
done