  #!/bin/bash

  # Do not forget to select a proper partition if
  # the default one is no fit for the job!

  #SBATCH --output=results/out.%j
  #SBATCH --error=results/err.%j
  #SBATCH --nodes=2          # number of nodes
  #SBATCH --ntasks=2         # number of processor cores (i.e. tasks)
  #SBATCH --tasks-per-node=1 # number of tasks per node
  #SBATCH --exclusive
  #SBATCH --time=00:10:00    # walltime

  # Good Idea to stop operation on first error.
  set -e

  # Load environment modules for your application here.
  source /etc/profile.d/modules.sh
  module load mpich
  module load gcc/11.2.0

  # Actual work starting here.
  srun ./hello_world_omp
