#!/bin/bash

USAGE="Usage: bash scripts/plot_throughput.sh LOGDIR MODE"

LOGDIR="$1"
MODE="$2"
# LOGDIR2="$3"

shift 2
while getopts "p:l2:a:" flag; do
  case "$flag" in
      p) PROJECT_DIR=$OPTARG;;
      l2) LOGDIR2=$OPTARG;;
      a) ADD_LEGEND=$OPTARG;;
      *) echo "Invalid option. ${USAGE}"
  esac
done

if [ -z "${LOGDIR}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${MODE}" ]
then
  echo "${USAGE}"
  exit 1
fi

SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"
if [ -n "$PROJECT_DIR" ]; then
  SINGULARITY_OPTS="$SINGULARITY_OPTS --bind ${PROJECT_DIR}:${PROJECT_DIR}"
fi
if [ "${MODE}" = "n_agents" ]
then
    singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
        python env_search/analysis/throughput_vs_n_agents.py \
            --logdirs_plot "$LOGDIR" \
            --add_legend "$ADD_LEGEND"
fi

if [ "${MODE}" = "thr_time" ]
then
    singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
        python env_search/analysis/throughput_thr_time.py \
            --logdirs_plot "$LOGDIR" \
            --add_legend "$ADD_LEGEND"
fi

if [ "${MODE}" = "cross_n_agents" ]
then
    singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
        python env_search/analysis/throughput_vs_n_agents_cross.py \
            --all_logdirs_plot "$LOGDIR" \
            --add_legend "$ADD_LEGEND"
fi

if [ "${MODE}" = "cross_thr_time" ]
then
    singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
        python env_search/analysis/throughput_thr_time_cross.py \
            --all_logdirs_plot "$LOGDIR" \
            --add_legend "$ADD_LEGEND"
fi

if [ "${MODE}" = "cross_thr_time_w_n_agents" ]
then
    singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
        python env_search/analysis/throughput_thr_time_w_n_agents_cross.py \
            --all_thr_time_logdirs_plot "$LOGDIR" \
            --all_n_agents_logdirs_plot "$LOGDIR2"
fi