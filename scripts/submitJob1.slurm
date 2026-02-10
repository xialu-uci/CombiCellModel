
#!/bin/bash

   #SBATCH --job-name=task42ci_modelF5c
   #SBATCH --partition=free          ## partition/queue name
   
   #SBATCH --cpus-per-task=4 ## (-c) number of cpus to allocate (PER ARRAY JOB)
   #SBATCH --mem=16G  # Use all available memory on the node
   #SBATCH --array=1-7 ## number of array tasks
   #SBATCH --ntasks=1 ## (-n) number of tasks to launch (PER ARRAY JOB)

   #SBATCH --output=/pub/stilese/2026/combiOut
   #SBATCH --time=0-00:10:00 # Maximum time. if this takes more then a minute something is horribly wrong
   # hpc3 environment
   module load julia-j

   # Sync NFS cache and check compiled packages exist
   sync
   echo "Checking compiled cache exists:"
   ls -la $JULIA_DEPOT_PATH/compiled/v1.12/CombiCellModelLearning/ 2>/dev/null || echo "Model8 not found in compiled cache!"
   echo "Cache file checksums (compare with instantiate output):"
   md5sum $JULIA_DEPOT_PATH/compiled/v1.12/CombiCellModelLearning/*.ji 2>/dev/null || echo "No .ji files found"

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
   SOURCEPATH=/pub/stilese/2026/CombiCellModel/
   OUTPUTPATH=/pub/stilese/2026/combiOut

   RUNNAME=taskA${SLURM_ARRAY_TASK_ID}

   cd $SOURCEPATH

   # echo "Updating julia using juliaup"
   # date; juliaup update; date

   INPUT=$SOURCEPATH/batch_scripts/args7.txt
   Args=$(awk "NR==$SLURM_ARRAY_TASK_ID" $INPUT)

   echo "Task 42ci: ModelF5c Corduroy Learning (depends on Task 0a)"
   echo "Arguments: $Args"

   # Test if Model8 loads without recompilation (should be <5 seconds if precompiled)
   echo "=== Testing Model8 load (precompilation check) ==="
   echo "Expected: <5 seconds if precompiled, >60 seconds if recompiling"
   date
   /usr/bin/time -v julia --project=Model8 -e 'using Model8; println("Model8 loaded successfully")' 2>&1 | grep -E "(User time|Elapsed|Precompiling|Model8 loaded)"
   date
   echo "=== End precompilation check ==="

   echo "Running learn_corduroy notebook"
   date; /usr/bin/time -v stdbuf -oL julia --project=Model8 notebooks/learn_corduroy.jl \
   configs/task42ci_modelF5c_corduroy.toml \
   $RUNNAME $Args \
   > $OUTPUTPATH/logs/$SLURM_ARRAY_JOB_ID.$RUNNAME.$SLURM_ARRAY_TASK_ID.log 2>&1 ; date

   # After job completes, move the logs
   logdate=$(date +%y%m%d)
   mkdir -p "$OUTPUTPATH/logs/$logdate"
   # Note: slurm .out/.err files are still open during job - cannot move them here
   # mv "$OUTPUTPATH/logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "$OUTPUTPATH/logs/$logdate/"
   # mv "$OUTPUTPATH/logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "$OUTPUTPATH/logs/$logdate/"
   mv "$OUTPUTPATH/logs/$SLURM_ARRAY_JOB_ID.$RUNNAME.$SLURM_ARRAY_TASK_ID.log" "$OUTPUTPATH/logs/$logdate/"

