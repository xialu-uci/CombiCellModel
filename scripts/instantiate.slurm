#!/bin/bash

#SBATCH --job-name=combicell_sim_fixedInt  ## job name
#SBATCH -p free              ## use free partition
#SBATCH --cpus-per-task=4 ## (-c) number of cpus to allocate (PER ARRAY JOB)
#SBATCH --mem=16G  # Use all available memory on the node
#SBATCH --array=1-1 ## number of array tasks
#SBATCH --ntasks=1 ## (-n) number of tasks to launch (PER ARRAY JOB)
#SBATCH --error=%x.%A.err    ## Slurm error  file, %x - job name, %A job id 
#SBATCH --out=%x.%A.out      ## Slurm output file, %x - job name, %A job id
#SBATCH --time=0-00:10:00 # Maximum time.

# hpc3 environment
module load julia-j

# Debug: Report environment for troubleshooting precompilation issues
echo "=== Environment Debug Info ==="
echo "Hostname: $(hostname)"
echo "Node CPU model: $(grep 'model name' /proc/cpuinfo | head -1)"
echo "JULIA_DEPOT_PATH: $JULIA_DEPOT_PATH"
echo "Julia version: $(julia --version)"
echo "Julia binary: $(which julia)"
julia -e 'println("Julia Sys.BINDIR: ", Sys.BINDIR)'
julia -e 'println("Julia DEPOT_PATH: ", DEPOT_PATH)'
julia -e 'println("Julia Base.julia_cmd(): ", Base.julia_cmd())'
echo "=== End Debug Info ==="

export JULIA_NUM_THREADS=4
julia -e 'println("Julia threads: ", Threads.nthreads()); println("CPU threads visible: ", Sys.CPU_THREADS)'

# Set up the experiment
SOURCEPATH=/pub/stilese/2026/CombiCellModel/CombiCellModelLearning 
OUTPUTPATH=/pub/stilese/2026/combiOut

cd $SOURCEPATH

# echo "Updating julia using juliaup"
# date; juliaup update; date

echo "Running Julia depot instantiation"
date; /usr/bin/time -v stdbuf -oL julia --project=CombiCellModelLearning -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); using CombiCellModelLearning' \
> $OUTPUTPATH/logs/$SLURM_ARRAY_JOB_ID.instantiate.$SLURM_ARRAY_TASK_ID.log 2>&1 ; date

# Verify compiled cache was created
echo "=== Verifying compiled cache ==="
echo "Compiled Model8 files:"
ls -la $JULIA_DEPOT_PATH/compiled/v1.12/CombiCellModelLearning/ 2>/dev/null || echo "WARNING: Model8 not in compiled cache!"
echo "Cache file checksums (save these to compare with batch jobs):"
md5sum $JULIA_DEPOT_PATH/compiled/v1.12/CombiCellModelLearning/*.ji 2>/dev/null || echo "No .ji files found"
echo "Sync to flush NFS cache:"
sync


# After job completes, move the logs
logdate=$(date +%y%m%d)
mkdir -p "$OUTPUTPATH/logs/$logdate"
# Note: slurm .out/.err files are still open during job - cannot move them here
# mv "$OUTPUTPATH/logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "$OUTPUTPATH/logs/$logdate/"
# mv "$OUTPUTPATH/logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "$OUTPUTPATH/logs/$logdate/"
mv "$OUTPUTPATH/logs/$SLURM_ARRAY_JOB_ID.instantiate.$SLURM_ARRAY_TASK_ID.log" "$OUTPUTPATH/logs/$logdate/"
