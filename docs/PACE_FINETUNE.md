# Fine-tuning LegalBERT on PACE

Everything needed to run this is already committed to the repo, so a
`git clone`/`git pull` on PACE is the only thing that has to "carry over" —
no chat context, no manual file copying required.

**Status: done.** This has been run end-to-end on PACE Phoenix. Results are
committed at `results/benchmark_20260728T234451Z.json` and
`results/presentation/` (charts + `summary.json`). The rest of this doc is
the runbook, kept for reproducing the run or fine-tuning again (e.g. with
more data, or after extending `data/legalbert_finetune_train.json` beyond
the current 7 categories).

**Note on which cluster:** the original attempt targeted PACE ICE, but its
login nodes closed the SSH connection immediately after password entry (no
Duo challenge ever appeared — see the troubleshooting note at the bottom if
this recurs). The run that actually succeeded used **PACE Phoenix**
(`login-phoenix.pace.gatech.edu`), which is what `scripts/finetune_legalbert.sbatch`
targets today.

## Results

Fine-tuned LegalBERT (`nlpaueb/legal-bert-base-uncased` + a linear
classification head, trained end-to-end via cross-entropy, 6 epochs) versus
the two backends already benchmarked since Checkpoint 2/3, all scored
against the same 350-clause `data/cuad_sample.json` fixture:

| Backend | Macro F1 |
|---|---|
| Fine-tuned LegalBERT | **0.966** |
| Rule-based (keyword) | 0.646 |
| Zero-shot LegalBERT (cosine similarity, untrained) | 0.554 |

See `results/presentation/` for the per-category breakdown and training
curves (6 epochs, 560 train / 140 val clauses).

**Important caveat, checked after the fact:** 78% of the benchmark's source
contracts also contributed other clauses to the training set — so while no
single clause's exact text is duplicated between train and benchmark, the
model has already seen the drafting style of most "held-out" documents.
0.966 is a real number but overstates generalization to genuinely unseen
contract templates. A separate, cleaner check — running the classifier on
two hand-written test contracts with zero relation to CUAD, in two very
different registers (plain English and dense legalese) — got it 13/13
correct on its trained categories, which is the more trustworthy signal of
real generalization.

**Known limitation:** the fine-tuned classifier only covers 7 of
ContractLens's 10 categories — CUAD has no ground truth for Confidentiality,
Indemnification, or Force Majeure (same gap `data/cuad_sample.json` has had
since Checkpoint 2). Unlike the rule-based classifier, it has no
"Unclassified" output — fed a clause from one of those 3 categories, it
picks the closest of its 7 known labels instead of abstaining (observed:
Confidentiality → Intellectual Property, Indemnification → Liability, both
at high confidence). This is why it's wired in as an opt-in backend
(`CONTRACTLENS_CLASSIFIER=legalbert-finetuned`), not the default.

## What's already done (in this repo)

- `data/legalbert_finetune_train.json` (560 clauses) / `legalbert_finetune_val.json`
  (140 clauses) — a balanced, 7-category train/val split pulled from the same
  public CUAD source dataset `scripts/generate_cuad_sample.py` uses, with
  every clause already in `data/cuad_sample.json` explicitly excluded (see
  `scripts/generate_finetune_dataset.py`'s docstring) — so fine-tuning never
  trains on the exact same clause text the benchmark later scores against
  (though see the document-level leakage caveat above).
- `scripts/finetune_legalbert.py` — the actual training script. Verified
  locally with `python -m scripts.finetune_legalbert --smoke` (a 24-example,
  1-epoch, CPU-only run) to confirm the whole pipeline runs end-to-end before
  you ever touch PACE.
- `scripts/finetune_legalbert.sbatch` — the SLURM job script for Phoenix.
- `app/clauses/classifier.py`'s `FineTunedLegalBertClassifier` — already
  wired into `get_classifier()` (`CONTRACTLENS_CLASSIFIER=legalbert-finetuned`)
  and into `scripts/evaluate_clauses.py` / `scripts/run_benchmarks.py`
  (`--classifier legalbert-finetuned`), so once you have a trained model
  there's nothing left to wire up — just point at it and run the benchmark.

## Step 1: SSH in and get the repo

```bash
ssh <your-gt-username>@login-phoenix.pace.gatech.edu
```

If PACE prompts for a password and then closes the connection with no Duo
challenge shown, `scp`/`ssh` may need SSH `ControlMaster` multiplexing to
render the Duo step properly — authenticate once in a plain interactive
`ssh` session first, leave it open, and subsequent connections (including
from an IDE's Remote-SSH) will reuse it instead of hitting the same issue.

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

`scripts/finetune_legalbert.sbatch` has placeholders (`--account`, `--qos`)
that are specific to your allocation — a wrong value fails at submit time,
not after burning a GPU-hour, so it's worth checking first:

```bash
# Your account name(s):
pace-quota
# or:
sacctmgr show associations user=$USER format=account,partition,qos

# Confirm the GPU partition exists and has capacity:
sinfo -p gpu-v100
```

Edit `scripts/finetune_legalbert.sbatch`, replacing `CHANGE_ME` (and the
partition/GPU type if your allocation doesn't have `gpu-v100`/V100s).

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
downloading `torch`), pins `torch==2.5.1`/`transformers==4.48.3` (Phoenix's
CUDA 12.1 wheels only go up to torch 2.5.x, and transformers' `.bin`-loading
guard for CVE-2025-32434 requires torch ≥2.6 unless pinned below the
guarded version — see the sbatch script's comments), then downloads
`nlpaueb/legal-bert-base-uncased` (~440MB, one-time) and trains for 6 epochs
over 560 examples. On a V100 this training step takes low single-digit
minutes, not hours — the `--time=01:00:00` ceiling is a safety margin, not
the expected duration.

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

# And the rule-based baseline for reference:
python -m scripts.evaluate_clauses --backend rule
```

(`--out ""` skips writing a new timestamped file while you're iterating;
drop it once you have a run you want to keep as evidence.)

To regenerate the presentation charts from a new run's numbers, update the
literal arrays at the top of `scripts/make_presentation_charts.py` (they're
copied from the job's `.out` log, not computed automatically) and run
`python -m scripts.make_presentation_charts` (requires `matplotlib`, listed
in `requirements.txt`).

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
scp -r <username>@login-phoenix.pace.gatech.edu:<path-to-repo>/ContractLens/models/legalbert-finetuned ./models/
```

If this hangs at the password prompt with no Duo challenge shown, see the
`ControlMaster` note in Step 1 — the same fix applies to `scp`.

Then locally: `CONTRACTLENS_CLASSIFIER=legalbert-finetuned uvicorn app.main:app --reload`.

## If something goes wrong

- **`ssh`/`scp` hangs or closes right after the password prompt, no Duo
  challenge shown**: see the `ControlMaster` fix in Step 1. This is a client
  limitation with rendering PACE's two-factor challenge over `scp`'s
  non-interactive protocol, not a PACE outage.
- **Job stays `PD` (pending) a long time**: the GPU partition may be busy;
  check `sinfo -p gpu-v100` for idle nodes, or try a different GPU type in
  the sbatch script if your allocation has one with a shorter queue.
- **`pip install` fails on a package**: Phoenix compute nodes usually do
  have internet access for pip/HuggingFace downloads, but if a proxy is
  required on your account, check your onboarding email/PACE docs for the
  module or env var that enables it.
- **Everything about job accounting fails**: run `pace-whoami` to confirm
  which allocations you're actually a member of.
