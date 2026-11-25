# Running on Hazel (NC State HPC)

This is the Hazel counterpart of `runs/modal_experiment.py`. It runs the same
experiment harness the same way: prepare the token cache once on CPU, then train
one config per H100. Hazel docs: https://hpc.ncsu.edu/GettingStarted/GettingStarted.php

Hazel facts these scripts rely on (from hpc.ncsu.edu, November 2026):

- The scheduler is **Slurm** (`sbatch`, `squeue`, `scancel`). LSF retires after the
  Fall semester, so the many `#BSUB` examples on the site are out of date.
- GPU jobs use partition `gpu`, and the GPU type is required: `--gres=gpu:h100:1`.
  The `gpu` QOS allows up to **4 days** per job. H100 nodes have 4 × 80 GB.
- **Nothing heavy runs on login nodes**; such processes are killed without notice.
  Installing packages there is fine, which is all `setup.sh` does.
- Storage:

  | Path | Size | Notes | Used for |
  |---|---|---|---|
  | `/home/$USER` | 15 GB | backed up | the git clone, `~/.config/bartholomew/secrets.env` |
  | `/usr/local/usrapps/$GROUP/$USER` | 100 GB per project | backed up, **read-only from compute nodes** | uv, Python, the venv |
  | `/share/$GROUP/$USER` | 20 TB per project | **not backed up; files unused for 30 days are deleted** | token caches, checkpoints, HF/W&B/compile caches, logs, frozen code |

  Durable results already go to Hugging Face and W&B, so the purge only costs a re-download.

## 1. Get access through the NAIL project (one time)

Hazel accounts are per faculty project. Professor Ditto (the PI) or whoever
manages the NAIL project adds you in the Research Services app:
https://oitrsvprd.oit.ncsu.edu → **Manage Project** → pencil icon next to the
project → add your **Unity ID** as the login. Steps:
https://hpc.ncsu.edu/Accounts/add.php

- If you are already in a different HPC project, this **moves** you out of it.
  If the app says the combination "has already been taken", email help@ncsu.edu
  and ask to be moved to the NAIL project.
- You also have to accept the Acceptable Use Policy: https://hpc.ncsu.edu/About/AUP.php
- Once you're added, `id -gn` on Hazel prints the project's group, and
  `/share/<group>` exists. The scripts find the group this way. If your login group
  is a different one, set `HAZEL_GROUP=<group>` in your shell before running them.
- Ask whether the project has `/usr/local/usrapps/<group>` space. If it doesn't,
  the venv falls back to `/share`, where the purge can delete it. The PI can
  request space: https://hpc.ncsu.edu/Software/Software.php#user

## 2. Log in

```bash
ssh <unity-id>@login.hpc.ncsu.edu
```

Every login asks for your password and Duo. To get one Duo prompt per work
session, add this to `~/.ssh/config` on your laptop, then use `ssh hazel`:

```text
Host hazel
    HostName login.hpc.ncsu.edu
    User <unity-id>
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 4h
```

You can also open a shell in the browser through Open OnDemand
(https://servood.hpc.ncsu.edu → Clusters → Hazel Shell Access).

## 3. Install (one time, and again after uv.lock changes)

On the login node:

```bash
git clone https://github.com/zachnorton14/bartholomew-v2.git ~/bartholomew-v2
cd ~/bartholomew-v2
git checkout engram-ablations          # or whichever branch you are running
bash runs/hazel/setup.sh
```

`setup.sh` installs uv and the exact `uv.lock` environment with Python 3.12 and
CUDA 12.8 torch 2.9.1, the same versions as the Modal image. It also caches the FA3
kernel and checks that the configs parse.

Then create the credentials file yourself. Every job reads it:

```bash
mkdir -p ~/.config/bartholomew && umask 077
cat > ~/.config/bartholomew/secrets.env <<'KEYS'
HF_TOKEN=hf_...
WANDB_API_KEY=...
KEYS
```

These are the same tokens as the Modal secrets. The HF token needs write access to
the config's `artifacts.repo`, and the W&B key needs access to its `wandb.entity`.

## 4. Check a GPU node

```bash
runs/hazel/submit.sh check
```

This is a 30-minute H100 job. It reports:

- **Whether compute nodes can reach huggingface.co and api.wandb.ai.** Hazel's docs
  don't say. The trainer needs both, because it uploads checkpoints to HF and logs
  to W&B. If they're unreachable, see [No internet on compute nodes](#no-internet-on-compute-nodes).
- Whether the HF and W&B credentials work.
- The `scripts.gpu_preflight` result: FP8, FA3, torch.compile and NCCL on the H100,
  plus the driver version. CUDA 12.8 torch needs driver 570 or newer.

## 5. Launch runs

```bash
runs/hazel/submit.sh launch --arm smoke --dry-run    # print the plan
runs/hazel/submit.sh launch --arm smoke              # 20-step end-to-end test
runs/hazel/submit.sh launch --arm baseline,baseline-s43,lite,values-32m
runs/hazel/submit.sh launch --config configs/base/<config>.json --eval bpb
runs/hazel/submit.sh status
```

What a launch does:

1. It refuses to run with uncommitted changes (`--allow-dirty` overrides). It then
   copies the commit to `/share/$GROUP/$USER/code/<sha>/`. Jobs run from that
   copy, so pulling or editing the repo while jobs wait in the queue doesn't change them.
2. It submits one CPU job (`compute` partition, 16 cores) per distinct token cache.
   That job restores the hosted cache or builds it from the dataset, so no GPU
   time is spent tokenizing.
3. It submits one H100 job per config. Each waits for its cache job to succeed,
   then runs the GPU preflight, links the cache, runs `scripts.experiment train`,
   and runs the eval if you asked for one.

Logs are in `/share/$GROUP/$USER/logs/<job-name>-<jobid>.out`. Cancel one job
with `scancel <jobid>`, or all of yours with `scancel -u $USER`.

Train jobs ask for 2 days by default. Raise that (the maximum is 4 days) or the
memory with `HAZEL_TRAIN_SBATCH_ARGS="--time=3-00:00:00 --mem=128G"`. The same
works for prepare jobs with `HAZEL_PREP_SBATCH_ARGS`. A job that hits the time
limit can be relaunched with the same config, and it resumes from its latest
complete checkpoint.

Useful Hazel commands: `si --gpus` (free GPUs by type), `sqos` (your QOS limits),
`seff <jobid>` (efficiency after a job ends), `quota_display`.

## No internet on compute nodes

If `submit.sh check` shows huggingface.co as unreachable, three things break:
building the cache in the prepare job, the HF uploads during training, and live
W&B logging. Hazel's `xfer` partition does have external access, but it is capped
at 4 cores and 24 GB. Before changing the harness, ask help@ncsu.edu whether GPU
nodes can get outbound HTTPS, or whether there is a proxy. Many clusters provide one.
