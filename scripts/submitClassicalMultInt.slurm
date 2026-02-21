#!/bin/bash

# the very first line is a special directive for Bash shell, do not remove
# lines that start with "#SBATCH" are special directives for Slurm
# other lines that start with "#" are comments

#SBATCH --job-name=combicell_sim_multInt  ## job name
#SBATCH -A stilese
#SBATCH --partition=standard  
#SBATCH --cpus-per-task=4 ## (-c) number of cpus to allocate (PER ARRAY JOB)
#SBATCH --mem=16G  # Use all available memory on the node
#SBATCH --array=1-4 ## number of array tasks # change for parellization
#SBATCH --ntasks=1 ## (-n) number of tasks to launch (PER ARRAY JOB)
#SBATCH --error=/pub/stilese/2026/CombiCellLocal/logs/slurm-%A.err    ## Slurm error  file, %x - job name, %A job id 
#SBATCH --out=/pub/stilese/2026/CombiCellLocal/logs/slurm-%A.out      ## Slurm output file, %x - job name, %A job id
#SBATCH --time=0-00:40:00 # Maximum time.

# error and out are default from the first job I ran?

# hpc3 environment
module load julia-1.12.5

# Sync NFS cache and check compiled packages exist
sync
echo "Checking compiled cache exists:"
ls -la $JULIA_DEPOT_PATH/compiled/v1.11/CombiCellModelLearning/ 2>/dev/null || echo "Model8 not found in compiled cache!"
echo "Cache file checksums (compare with instantiate output):"
md5sum $JULIA_DEPOT_PATH/compiled/v1.11/CombiCellModelLearning/*.ji 2>/dev/null || echo "No .ji files found"

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
julia -e 'using LinearAlgebra; println("BLAS threads: ", BLAS.get_num_threads())'

# Set up the experiment
SOURCEPATH=/pub/stilese/2026/CombiCellModel/ # don't think i need this for this run
OUTPUTPATH=../CombiCellLocal/experiments/0220_realData_testParallel1/


echo "Test1: does parallelized classicalMultInt.jl run on hpc3?"

# Test if Model8 loads without recompilation (should be <5 seconds if precompiled)
echo "=== Testing Model8 load (precompilation check) ==="
echo "Expected: <5 seconds if precompiled, >60 seconds if recompiling"
date
/usr/bin/time -v julia +1.11.6 --project=CombiCellModelLearning -e 'using CombiCellModelLearning; println("CombiCellModelLearning loaded successfully")' 2>&1 | grep -E "(User time|Elapsed|Precompiling|CombiCellModelLearning loaded)"
date
echo "=== End precompilation check ==="

# execute script
INPUT=$SOURCEPATH/CombiCellModelLearning/scripts/144args.txt
ARGS=$(awk "NR==$SLURM_ARRAY_TASK_ID" $INPUT)

echo "Running classicalMultInt"
date; /usr/bin/time -v stdbuf -oL julia --project=CombiCellModelLearning CombiCellModelLearning/src/classicalMultInt.jl $ARGS

# After job completes, move the logs
logdate=$(date +%y%m%d)
mkdir -p "$OUTPUTPATH/logs/$logdate"
# Note: slurm .out/.err files are still open during job - cannot move them here
mv "$OUTPUTPATH/logs/$SLURM_ARRAY_JOB_ID.$RUNNAME.$SLURM_ARRAY_TASK_ID.log" "$OUTPUTPATH/logs/$logdate/"
