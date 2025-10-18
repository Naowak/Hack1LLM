import argparse
import datetime
import os
import subprocess
import random

parser = argparse.ArgumentParser(description="Plafrim SLURM launcher")

# ---- SLURM options ----
parser.add_argument(
    "--node", type=str, default=None, help="Node to use (e.g., sirocco)"
)
parser.add_argument(
    "--cpu", action=argparse.BooleanOptionalAction, help="Use CPU partition"
)
parser.add_argument(
    "--v100", action=argparse.BooleanOptionalAction, help="Use V100 GPUs"
)
parser.add_argument(
    "--a100", action=argparse.BooleanOptionalAction, help="Use A100 GPUs"
)
parser.add_argument(
    "--dev", action=argparse.BooleanOptionalAction, help="Development mode"
)
parser.add_argument("--cpus-per-task", type=int, default=1, help="CPUs per task")
parser.add_argument("--n_gpu", type=int, default=0, help="GPUs to request")
parser.add_argument("--hour", type=int, default=20, help="Job duration in hours")

args = parser.parse_args()

# ---- Options ----


def generate_slurm_script(args, job_name):
    
    # ====== Configuration ======
    seed = random.randint(0, 10**4)
    run_name = None
    # ===========================
    
    # Define log directory and file
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if run_name is None:
        run_name = f"{date}_PLA{seed}"
    log_dir = f"./logs/{run_name}"
    log_file = f"{log_dir}/log.txt"
    os.makedirs(log_dir, exist_ok=True)

    list_lines_script = ["#!/bin/bash"]
    list_lines_modules = []

    # Set the time limit based on the mode
    if args.dev:
        hour = 2
    else:
        hour = args.hour

    # Partition, GPU, and CPU selection
    list_constraint = []
    if args.node is None:
        if args.cpu:
            args.n_gpu = 0
        elif args.v100:
            list_constraint = ["sirocco", "v100"]
            n_cpu = min(args.n_gpu * 10, 40)
        elif args.a100:
            list_constraint = ["sirocco", "a100"]
            n_cpu = min(args.n_gpu * 10, 40)
        else:
            raise ValueError("Please specify --cpu, --v100, or --a100")
    else:
        list_constraint = [args.node]
        if args.cpu:
            args.n_gpu = 0
        if args.v100:
            list_constraint.append("v100")
            n_cpu = min(args.n_gpu * 10, 40)
        elif args.a100:
            list_constraint.append("a100")
            n_cpu = min(args.n_gpu * 10, 40)
        else:
            raise ValueError("Please specify --cpu, --v100, or --a100")
    if len(list_constraint) > 0:
        list_lines_script.append(f"#SBATCH -C {'&'.join(list_constraint)}")

    if args.cpus_per_task:
        n_cpu = args.cpus_per_task

    # Environment setup
    list_lines_env = [
        "module load build/conda/4.10",
        "conda activate /home/hack-gen1/vllm_venv/",
        "module load tools/git/2.36.0 compiler/gcc/11.2.0",
        "module load compiler/gcc/11.2.0",
    ]

    # Main script to run
    script_main = "echo 'Hello World!'"
    
    # Create the SLURM script
    script_begin = "\n".join(list_lines_script)
    script_module = "\n".join(list_lines_modules)
    script_env = "\n".join(list_lines_env)

    script = f"""#!/bin/bash
{script_begin}
#SBATCH --job-name=mg{seed}
#SBATCH -p hack                     # same partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
## SBATCH --gres=gpu:{args.n_gpu}    # i don't really know, this with 1 or more raise an error, and torch.cuda.device_count() result seems independent of this value
#SBATCH --cpus-per-task={n_cpu}
#SBATCH --hint=nomultithread
#SBATCH --time={hour}:00:00
#SBATCH --output=./logs/{run_name}/log_outerr.txt
#SBATCH --error=./logs/{run_name}/log_outerr.txt
#SBATCH --exclusive                  # optional, matches salloc

{script_module}

{script_env}

cd $HOME/hack1llm
mkdir -p {log_dir}
echo "SLURM_NODENAME: $SLURM_NODENAME"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "hostname -I: $(hostname -I)"

python scripts/check_cuda.py > {log_file} 2>&1
{script_main} >> {log_file} 2>&1
echo "Finished at $(date)" >> {log_file} 2>&1
"""
    with open(f"{log_dir}/script.slurm", "w") as f:
        f.write(script)
    print(f"Generated SLURM script for run {run_name}.")
    print(f"Sync command : syncr {log_dir} pla:/home/tboulet/hack1llm")
    print(f"Log file     : {log_file}")
    return script


# Create the SLURM script and submit it
slurmfile_path = "temp_job.slurm"
full_script = generate_slurm_script(args, "hack1llm_job")

with open(slurmfile_path, "w") as f:
    f.write(full_script)

subprocess.call(f"sbatch {slurmfile_path}", shell=True)
os.remove(slurmfile_path)
