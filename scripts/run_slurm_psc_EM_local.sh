print_header() {
  echo
  echo "------------- $1 -------------"
}

CONFIG="$1"
NUM_WORKERS="$2"
SEED="$3"
shift 3
DRY_RUN=""
RELOAD_ARG=""
while getopts "dr:" opt; do
  case $opt in
    d)
      echo "Using DRY RUN"
      DRY_RUN="1"
      ;;
    r)
      echo "Using RELOAD: $OPTARG"
      RELOAD_ARG="-r $OPTARG"
      ;;
  esac
done

DATE="$(date +'%Y-%m-%d_%H-%M-%S')"
LOGDIR="slurm_logs/slurm_${DATE}"
echo "SLURM Log directory: ${LOGDIR}"
mkdir -p "$LOGDIR"
SEARCH_SCRIPT="$LOGDIR/search.slurm"
SEARCH_OUT="$LOGDIR/search.out"

echo "\
#!/bin/bash
#SBATCH --job-name=EM_search_${DATE}
#SBATCH -N 1
#SBATCH -p EM
#SBATCH -t 24:00:00
#SBATCH -n $NUM_WORKERS
#SBATCH --account=cis220074p
#SBATCH --output $SEARCH_OUT
#SBATCH --error $SEARCH_OUT

echo
echo \"========== Start ==========\"
date

bash scripts/run_local.sh $CONFIG 42 $NUM_WORKERS $RELOAD_ARG

echo
echo \"========== Done ==========\"

date" >"$SEARCH_SCRIPT"

# Submit the search script.
if [ -z "$DRY_RUN" ]; then sbatch "$SEARCH_SCRIPT"; fi


print_header "Monitoring Instructions"
echo "\
To view output from the search and main script, run:

  tail -f $SEARCH_OUT
"

#
# Print cancellation instructions.
#

# if [ -n "$DRY_RUN" ]
# then
#   print_header "Skipping cancellation, dashboard, postprocessing instructions"
#   exit 0
# fi

# # Record job ids in logging directory. This can be picked up by
# # scripts/slurm_cancel.sh in order to cancel the job.
# echo -n -e "$JOB_IDS" > "${LOGDIR}/job_ids.txt"

# print_header "Canceling"
# echo "\
# To cancel this job, run:

#   bash scripts/slurm_cancel.sh $LOGDIR
# "