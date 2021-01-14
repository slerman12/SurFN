import argparse
import subprocess


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="name")
parser.add_argument('--cpu', action='store_true', default=False,
                    help='uses CPUs')
parser.add_argument('--bigger-gpu', action='store_true', default=False,
                    help='uses K80 GPU')
parser.add_argument('--biggest-gpu', action='store_true', default=False,
                    help='uses V100 GPU')
parser.add_argument('--file', type=str, default="../train_sweep.py")
parser.add_argument('--params', type=str, default="")
parser.add_argument('--module', type=str, default="anaconda3/2020.07")
args = parser.parse_args()


def slurm_script_generalized():
    return r"""#!/bin/bash
#SBATCH {}
#SBATCH -p csxu -P csxu {}
#SBATCH -t 5-00:00:00 -o ./{}.log -J {}
#SBATCH --mem=10gb 
{}
module load {}
python3 {} {}
""".format("-c 1" if args.cpu else "-p gpu", "" if args.cpu else "--gres=gpu", args.name, args.name,
           "#SBATCH -C K80" if args.bigger_gpu else "#SBATCH -C V100" if args.biggest_gpu else "",
           args.module, args.file, args.params)


with open("sbatch_script", "w") as file:
    file.write(slurm_script_generalized())
subprocess.call(['sbatch {}'.format("sbatch_script")], shell=True)
