import argparse
import datetime
import os
import random
import subprocess

import numpy as np

parser = argparse.ArgumentParser(description="Jean-Zay SLURM launcher")

# ---- SLURM options ----
parser.add_argument(
    "--cpus-per-task", type=int, default=None, help="Number of CPUs per task"
)
parser.add_argument(
    "--dev", action=argparse.BooleanOptionalAction, help="Development mode"
)
parser.add_argument("--hour", type=int, default=20)


args = parser.parse_args()

# ---- SWE-grid options ----


def generate_slurm_script(args, job_name):
    # Create run name and log_file
    seed = random.randint(0, 10**4)
    run_name = None
    
    # Create the log file name
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if run_name is None:
        run_name = f"{date}_JZ{seed}"
    log_dir = f"./logs/{run_name}"
    log_file = f"{log_dir}/log.txt"
    os.makedirs(log_dir, exist_ok=True)
    
    list_lines_script = ["#!/bin/bash"]
    list_lines_env = ["export TMPDIR=$JOBSCRATCH", "module purge"]
    
    # Set the time limit based on the mode
    if args.dev:
        hour = 2
    else:
        hour = args.hour

    # For each partition, specify account and constrains (if any), qos, and number of CPUs (if applicable)

    n_gpu = 1
    if True:
        list_lines_script.append("#SBATCH --account=imi@h100")
        list_lines_script.append("#SBATCH -C h100")
        n_cpu = min(int(n_gpu * 24), 96)
        if args.dev:
            list_lines_script.append("#SBATCH --qos=qos_gpu_h100-dev")
        elif hour <= 20:
            list_lines_script.append("#SBATCH --qos=qos_gpu_h100-t3")
        else:
            list_lines_script.append("#SBATCH --qos=qos_gpu_h100-t4")
        list_lines_env.append("module load arch/h100")
        
    if args.cpus_per_task:
        n_cpu = args.cpus_per_task

    
    
    # Ressources
    list_lines_script.append(f"#SBATCH --job-name=mg{seed}")
    list_lines_script.append(f"#SBATCH --nodes=1")
    list_lines_script.append(f"#SBATCH --ntasks-per-node=1")
    # list_lines_script.append(f"#SBATCH --gres={dict_ressources['gres']}")
    list_lines_script.append(f"#SBATCH --cpus-per-task={n_cpu}")
    list_lines_script.append("#SBATCH --hint=nomultithread")
    list_lines_script.append(f"#SBATCH --time={hour}:00:00")
    list_lines_script.append(f"#SBATCH --output=./logs/{run_name}/log_outerr.txt")
    list_lines_script.append(f"#SBATCH --error=./logs/{run_name}/log_outerr.txt")
    script_begin = "\n".join(list_lines_script)
    
    # Environment setup
    list_lines_env = [
        "source /lustre/fsn1/projects/rech/imi/upb99ot/venv/bin/activate",
    ]
    script_env = "\n".join(list_lines_env)
    
    
    # Main script to run
    list_lines_main = [
        f"python scripts/ftj.py --log_dir {log_dir}",
    ]
    script_main = "\n".join(list_lines_main)
    
    
    script = f"""{script_begin}

{script_env}


{script_main} >> {log_file} 2>&1
echo "Finished at $(date)" >> {log_file} 2>&1
"""
    with open(f"{log_dir}/script.slurm", "w") as f:
        f.write(script)
    print(f"Generated SLURM script for run {run_name}.")
    print(f"Sync command : syncr {log_dir} jz:/lustre/fswork/projects/rech/imi/upb99ot/Hack1LLM")
    print(f"Log file     : {log_file}")
    return script


# Create the SLURM script and submit it
slurmfile_path = "temp_job.slurm"
full_script = generate_slurm_script(args, "hack1llm_job")

with open(slurmfile_path, "w") as f:
    f.write(full_script)

subprocess.call(f"sbatch {slurmfile_path}", shell=True)
os.remove(slurmfile_path)


