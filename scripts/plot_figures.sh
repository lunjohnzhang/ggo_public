#!/bin/bash

USAGE="Usage: bash scripts/plot_figures.sh LOGDIR"

MANIFEST="$1"
MODE="$2"

shift 2
while getopts "p:" flag; do
  case "$flag" in
      p) PROJECT_DIR=$OPTARG;;
      *) echo "Invalid option. ${USAGE}"
  esac
done

collect="collect"
comparison="comparison"


if [ -z "${MANIFEST}" ];
then
  echo "${USAGE}"
  exit 1
fi

SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"
if [ -n "$PROJECT_DIR" ]; then
  SINGULARITY_OPTS="$SINGULARITY_OPTS --bind ${PROJECT_DIR}:${PROJECT_DIR}"
fi
if [ "${MODE}" = "${collect}" ];
then
    singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
        python env_search/analysis/figures.py \
            collect \
            --reps 1 "$MANIFEST"
fi

if [ "${MODE}" = "${comparison}" ];
then
    singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
        python env_search/analysis/figures.py \
            comparison \
            "figure_data.json"
fi