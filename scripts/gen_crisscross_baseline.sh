#!/bin/bash

USAGE="Usage: bash scripts/gen_crisscross_baseline.sh MAP_FILEPATH STORE_DIR DOMAIN"

MAP_FILEPATH="$1"
STORE_DIR="$2"
DOMAIN="$3"

shift 3
while getopts "p:b:" flag; do
  case "$flag" in
      p) PROJECT_DIR=$OPTARG;;
      b) BI_DIRECTED=$OPTARG;;
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
if [ -z "$BI_DIRECTED" ]; then
  BI_DIRECTED=True
fi
singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
    python env_search/analysis/gen_crisscross_baseline.py \
        --map_filepath "$MAP_FILEPATH" \
        --store_dir "$STORE_DIR" \
        --domain "$DOMAIN" \
        --bi_directed "$BI_DIRECTED"