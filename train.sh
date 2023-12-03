#!/bin/bash

#SBATCH --job-name='CGAN'
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=10
#SBATCH --time=1-0:0
# time format: <days>-<hours>:<minutes>

# Path to container
container="/data/containers/msoe-tensorflow-23.05-tf2-py3.sif"

# install tensorflow
install_tf_command="pip install -r requirements.txt"
# Command to run inside container
command="python main.py"

# Start a shell session within the container
singularity shell --nv -B /data:/data ${container} << EOF

${install_tf_command}
# Execute command inside container
${command}

# Exit the shell session
exit

EOF
