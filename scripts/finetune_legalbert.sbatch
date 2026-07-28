#!/bin/bash
# SLURM job for scripts/finetune_legalbert.py on Georgia Tech PACE's Phoenix cluster.
#
# Before your first real run, set --account / --partition / --qos to match
# your allocation (see docs/PACE_FINETUNE.md). Wrong values fail at submit time.

#SBATCH --job-name=legalbert-finetune
#SBATCH --account=CHANGE_ME                 # `pace-quota` or `sacctmgr show associations user=$USER`
#SBATCH --partition=gpu-v100                # confirm with `sinfo -p gpu-v100`
#SBATCH --qos=CHANGE_ME                     # e.g. inferno / embers — check your associations
#SBATCH --gres=gpu:V100:1                   # V100 is enough; A100 also works if available
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00                     # pip reinstall + training; ceiling protects against hangs
#SBATCH --output=finetune_legalbert_%j.out
#SBATCH --error=finetune_legalbert_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# Phoenix provides Anaconda via modules; if this module name doesn't exist on
# your account, run `module avail anaconda` or `module avail python` to find
# the right one and swap it in.
module load anaconda3
module load cuda/12.1 2>/dev/null || true

if [[ ! -d .venv-pace ]]; then
  python -m venv .venv-pace
fi
source .venv-pace/bin/activate
pip install --upgrade pip --quiet

# Phoenix cu121 wheels only go up to torch 2.5.x; pin transformers before the
# torch>=2.6 .bin-loading guard (CVE-2025-32434) so LegalBERT loads work.
pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu121 --quiet --force-reinstall
pip install -r requirements.txt --quiet
pip install 'transformers==4.48.3' --quiet --force-reinstall
pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu121 --quiet --force-reinstall

echo "=== GPU visible to this job ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU visible')"

echo "=== Starting fine-tune ==="
python -m scripts.finetune_legalbert --epochs 6 --batch-size 16

echo "=== Done. Fine-tuned model saved under models/legalbert-finetuned/final ==="
