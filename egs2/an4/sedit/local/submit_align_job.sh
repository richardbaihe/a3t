JOB_NAME=alignment

COM="srun python local/align_english.py"


sbatch --job-name "${JOB_NAME}" --partition CPUx40 \
    --wrap " $COM " 