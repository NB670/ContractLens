#!/bin/bash
# SLURM job for scripts/finetune_legalbert.py on Georgia Tech PACE's ICE cluster.
#
# Before your first real run, verify the placeholders below against your own
# account (see docs/PACE_FINETUNE.md for exactly how) -- ICE partition/QOS/
# account names are allocation-specific and a wrong value here fails the job
# at submission, not mid-run, so it's cheap to get right first.

#SBATCH --job-name=legalbert-finetune
#SBATCH --account=CHANGE_ME               # `pace-quota` or `sacctmgr show associations user=$USER` lists yours
#SBATCH --partition=ice-epyc-gpu          # ICE's GPU queue name as of this writing -- confirm with `sinfo -p ice-epyc-gpu`
#SBATCH --qos=ice-epyc-gpu
#SBATCH --gres=gpu:A100:1                 # or gpu:A40:1 if A100 has a long queue -- either is enough for this job
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00                   # the real run (560 train / 140 val, 6 epochs) finishes in minutes on a GPU;
                                           # this ceiling just protects against a hung job, not an expected duration
#SBATCH --output=finetune_legalbert_%j.out
#SBATCH --error=finetune_legalbert_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# ICE provides Anaconda via modules; if this module name doesn't exist on
# your account, run `module avail anaconda` or `module avail python` to find
# the right one and swap it in.
module load anaconda3

python -m venv .venv-pace
source .venv-pace/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

echo "=== GPU visible to this job ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU visible')"

echo "=== Starting fine-tune ==="
python -m scripts.finetune_legalbert --epochs 6 --batch-size 16

echo "=== Done. Fine-tuned model saved under models/legalbert-finetuned/final ==="
