#!/bin/bash

USAGE="Usage: bash scripts/plot_heatmap.sh LOGDIR MODE DOMAIN TRANSPOSE_BCS HEATMAP_ONLY"

LOGDIR="$1"
MODE="$2"
DOMAIN="$3"

shift 3
while getopts "p:t:h:" flag; do
  case "$flag" in
      p) PROJECT_DIR=$OPTARG;;
      t) TRANSPOSE_BCS=$OPTARG;;
      h) HEATMAP_ONLY=$OPTARG;;
      *) echo "Invalid option. ${USAGE}"
  esac
done

if [ -z "${LOGDIR}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${TRANSPOSE_BCS}" ]
then
  TRANSPOSE_BCS="True"
fi

if [ -z "${HEATMAP_ONLY}" ]
then
  HEATMAP_ONLY="False"
fi

SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"
if [ -n "$PROJECT_DIR" ]; then
  SINGULARITY_OPTS="$SINGULARITY_OPTS --bind ${PROJECT_DIR}:${PROJECT_DIR}"
fi

if [ "${DOMAIN}" = "kiva" ]
then
singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
    python env_search/analysis/heatmap.py \
        --logdir "$LOGDIR" \
        --mode "$MODE" \
        --kiva \
        --heatmap_only "$HEATMAP_ONLY" \
        --transpose_bcs "$TRANSPOSE_BCS"
fi

if [ "${DOMAIN}" = "competition" ]
then
singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
    python env_search/analysis/heatmap.py \
        --logdir "$LOGDIR" \
        --mode "$MODE" \
        --competition \
        --heatmap_only "$HEATMAP_ONLY" \
        --transpose_bcs "$TRANSPOSE_BCS"
fi