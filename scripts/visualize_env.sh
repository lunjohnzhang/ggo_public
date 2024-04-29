#!/bin/bash

USAGE="Usage: bash scripts/visualize_env.sh MAP_FILEPATH STORE_DIR DOMAIN"

MAP_FILEPATH="$1"
STORE_DIR="$2"
DOMAIN="$3"

shift 3
while getopts "p:" flag; do
  case "$flag" in
      p) PROJECT_DIR=$OPTARG;;
      *) echo "Invalid option. ${USAGE}"
  esac
done

if [ -z "${MAP_FILEPATH}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${STORE_DIR}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${DOMAIN}" ]
then
  echo "${USAGE}"
  exit 1
fi

SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"
if [ -n "$PROJECT_DIR" ]; then
  SINGULARITY_OPTS="$SINGULARITY_OPTS --bind ${PROJECT_DIR}:${PROJECT_DIR}"
fi
singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
    python env_search/analysis/visualize_env.py \
        --map_filepath "$MAP_FILEPATH" \
        --store_dir "$STORE_DIR" \
        --domain "$DOMAIN"