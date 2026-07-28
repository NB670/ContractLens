# Fine-tuning LegalBERT on PACE ICE

Everything needed to run this is already committed to the repo, so a
`git clone`/`git pull` on PACE is the only thing that has to "carry over" —
no chat context, no manual file copying required.

## What's already done (in this repo)

- `data/legalbert_finetune_train.json` (560 clauses) / `legalbert_finetune_val.json`
  (140 clauses) — a balanced, 7-category train/val split pulled from the same
  public CUAD source dataset `scripts/generate_cuad_sample.py` uses, with
  every clause already in `data/cuad_sample.json` explicitly excluded (see
  `scripts/generate_finetune_dataset.py`'s docstring) — so fine-tuning never
  trains on the clauses the benchmark later scores against.
- `scripts/finetune_legalbert.py` — the actual training script. Verified
  locally with `python -m scripts.finetune_legalbert --smoke` (a 24-example,
  1-epoch, CPU-only run) to confirm the whole pipeline runs end-to-end before
  you ever touch PACE.
- `scripts/finetune_legalbert.sbatch` — the SLURM job script for ICE.
- `app/clauses/classifier.py`'s `FineTunedLegalBertClassifier` — already
  wired into `get_classifier()` (`CONTRACTLENS_CLASSIFIER=legalbert-finetuned`)
  and into `scripts/evaluate_clauses.py` / `scripts/run_benchmarks.py`
  (`--classifier legalbert-finetuned`), so once you have a trained model
  there's nothing left to wire up — just point at it and run the benchmark.

## Step 1: SSH in and get the repo

```bash
ssh <your-gt-username>@ice.pace.gatech.edu
```

If this is your first time on this repo from PACE, clone it (use whatever
auth you already use for GitHub — a personal access token over HTTPS is
usually simplest on a cluster with no SSH agent forwarding set up):

```bash
git clone https://github.com/NB670/ContractLens.git
cd ContractLens
```

If you've cloned it before, just:

```bash
cd ContractLens
git pull
```

## Step 2: Confirm your actual account/partition/QOS names

`scripts/finetune_legalbert.sbatch` has three placeholders
(`--account`, `--partition`, `--qos`) that are specific to your allocation —
get a job submission error from a wrong value here, not a wasted GPU-hour, so
it's worth 2 minutes to check first:

```bash
# Your account name(s):
pace-quota
# or:
sacctmgr show associations user=$USER format=account,partition,qos

# Confirm the GPU partition exists and has capacity:
sinfo -p ice-epyc-gpu
```

Edit `scripts/finetune_legalbert.sbatch`, replacing `CHANGE_ME` (and the
partition/qos if `sinfo`/`sacctmgr` showed different names than
`ice-epyc-gpu`).

## Step 3: Submit the job

```bash
sbatch scripts/finetune_legalbert.sbatch
squeue -u $USER          # watch it move from PD (pending) to R (running)
```

Once running, tail the output:

```bash
tail -f finetune_legalbert_<jobid>.out
```

**What to expect:** the job creates a fresh venv and `pip install`s
`requirements.txt` (this is the slowest part — several minutes, dominated by
downloading `torch`), then downloads `nlpaueb/legal-bert-base-uncased`
(~440MB, one-time), then trains for 6 epochs over 560 examples — on any
ICE GPU (A40 or A100) this training step itself should take low single-digit
minutes, not hours. The `--time=01:00:00` ceiling in the sbatch script is
a safety margin, not the expected duration.

## Step 4: Check the result

```bash
cat models/legalbert-finetuned/final/eval_metrics.json
```

This is the held-out validation performance (macro precision/recall/F1 on
the 140-clause val split) — a first signal, but the real comparison is the
next step, which scores against `data/cuad_sample.json` (the same fixture
every other classifier in this project is already benchmarked against):

```bash
# Fine-tuned model:
python -m scripts.run_benchmarks --classifier legalbert-finetuned --retrieval-backend hashing --out ""

# For a true before/after on the *same model family*, also run the
# untrained zero-shot LegalBERT baseline:
python -m scripts.run_benchmarks --classifier legalbert --retrieval-backend hashing --out ""

# And the already-committed baseline for reference (results/benchmark_*.json
# in the repo was run with --classifier rule):
python -m scripts.evaluate_clauses --backend both
```

(`--out ""` skips writing a new timestamped file while you're iterating;
drop it once you have a run you want to keep as evidence.)

## Step 5: Get the result back to your laptop

The fine-tuned weights themselves (`models/legalbert-finetuned/`) are
gitignored on purpose (they're a few hundred MB of binary weights, not
source) — don't try to commit them. Two options:

**A. Just keep the numbers, not the weights** (simplest, and enough for a
report): copy `eval_metrics.json` and a `run_benchmarks --out <path>.json`
result back, or just paste the printed report into your notes. Commit that
results JSON from your laptop like any other `results/` artifact.

**B. Also bring the weights back**, if you want to actually run the
fine-tuned classifier locally later:

```bash
# from your laptop, not PACE:
scp -r <username>@ice.pace.gatech.edu:~/ContractLens/models/legalbert-finetuned ./models/
```

Then locally: `CONTRACTLENS_CLASSIFIER=legalbert-finetuned uvicorn app.main:app --reload`.

## If something goes wrong

- **Job stays `PD` (pending) a long time**: the GPU partition may be busy;
  check `sinfo -p ice-epyc-gpu` for idle nodes, or try `gpu:A40:1` instead of
  `gpu:A100:1` in the sbatch script (A40 is usually less contended).
- **`pip install` fails on a package**: ICE compute nodes usually do have
  internet access for pip/HuggingFace downloads, but if a proxy is required
  on your account, check your onboarding email/PACE docs for the module or
  env var that enables it.
- **Everything about job accounting fails**: run `pace-whoami` to confirm
  which allocations you're actually a member of.
