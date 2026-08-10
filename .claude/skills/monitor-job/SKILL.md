---
name: monitor-job
description: Babysit SLURM training jobs on the betty cluster cheaply, escalating to a Fable subagent only when something goes wrong. Trigger on "/monitor-job <jobid>", "monitor job", "babysit job/run", "watch my slurm job", or after submitting an sbatch when the user asks to keep an eye on it. Intended to run in an Opus main session (`claude --model opus`), so high-volume log context accrues at Opus prices while Fable is reserved for curated escalations.
---

# Job Babysitter — Opus monitor, Fable escalation

## Cost architecture (why this skill exists)

Three tiers. You are the middle one. Keeping each tier in its lane is the whole point:

- **Tier 0 — shell (free):** scripts on betty grep the raw logs. Only signal lines ever become events.
- **Tier 1 — you (Opus, cheap):** triage signals, read *targeted* log regions, judge soft anomalies, assemble handoffs. You absorb the high-volume context so Fable never has to.
- **Tier 2 — fable-debugger agent (expensive):** root-cause, fix, resubmit. Invoked only with a curated handoff file, never with raw logs, never for questions you can answer yourself.

Never cat a whole log into your context. Read line ranges (`sed -n 'A,Bp'`) around signals.

## Cluster facts

- Host: `ssh betty` (ProxyJump is configured in `~/.ssh/config`; add `-o ConnectTimeout=10` in scripts).
- Logs: `$ALELAB_DRIVE/GREP-PRISM/slurm-<jobid>.out` — array jobs use `slurm-<ArrayJobID>_<TaskID>.out`. `$ALELAB_DRIVE=/vast/projects/aribeiro/alelab/jporras`.
- Cluster repo: `~/sourcecode/GREP`. Code reaches the cluster via `git pull` only, never rsync.
- wandb: entity `alelab`, project `GREP-PRISM`.
- sbatch sources live in `scripts/*.sbatch` in this repo.

**Session setup note:** this skill runs unattended, so permission prompts stall it. If prompts appear for `ssh betty ...`, suggest the user allowlist `Bash(ssh betty*)` for the session (or run /fewer-permission-prompts) before arming the watcher.

## Inputs

`/monitor-job <jobid> [<jobid> ...]` — or `all` (everything in `squeue -u jporras`). Optional poll-interval override (default 5 min).

## Step 1 — Intake (once per job)

1. `ssh betty 'squeue -h -j <id> -o "%T %N %V %o"'` and `scontrol show job <id>` → state, node(s), script path, whether it's an array.
2. Find the sbatch script in local `scripts/`, read it. Record: config name (`--config-name=`), CLI overrides, log path.
3. `head -100` of the log on betty → extract the wandb run URL/id if printed, and note which metric keys the run logs (loss, grad_norm, steps/s, eval acc) for later health checks.
4. Write a state file `<scratchpad>/monitor-state-<jobid>.md`: job facts, escalation count (starts 0), direct-resubmit count (starts 0), prior escalation reports (empty). This file is the lineage memory — it follows the job across resubmits.

## Step 2 — Arm the watcher (Tier 0)

One **persistent Monitor** per job. Template — adapt log path, interval, and signature list to what intake found:

```bash
JOB=<jobid>; LOG=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-${JOB}.out
INTERVAL=300; STALL=1800; off=0; last_growth=$(date +%s); prev_state=""; tick=0
while true; do
  state=$(ssh -o ConnectTimeout=10 betty "squeue -h -j $JOB -o %T 2>/dev/null"); rc=$?
  # ssh itself failing exits 255; squeue on an aged-out job exits 1 with empty output
  [ $rc -eq 255 ] && state=SSH_FAIL
  [ "$state" != "$prev_state" ] && echo "STATE $JOB: ${prev_state:-none} -> ${state:-GONE}"
  prev_state=$state
  [ "$state" = "SSH_FAIL" ] && { sleep $INTERVAL; continue; }
  if [ -z "$state" ]; then
    ssh betty "sacct -j $JOB -o JobID,State,ExitCode,Elapsed -n" 2>/dev/null || true
    echo "TERMINAL $JOB: left queue"; break
  fi
  size=$(ssh betty "wc -c < $LOG" 2>/dev/null || echo "$off")
  if [ "$size" -gt "$off" ]; then
    ssh betty "tail -c +$((off+1)) $LOG" 2>/dev/null \
      | grep -nE "Traceback|CUDA (out of memory|error)|NCCL|RuntimeError|ValueError|AssertionError|slurmstepd: error|srun: error|Killed|Segmentation|torch.OutOfMemory|loss[=: ]+(nan|inf)|grad_norm[=: ]+(nan|inf|0\.0+([^0-9]|$))" \
      | head -20 | sed "s/^/SIG $JOB: /"
    off=$size; last_growth=$(date +%s)
  elif [ "$state" = "RUNNING" ] && [ $(( $(date +%s) - last_growth )) -gt $STALL ]; then
    echo "STALL $JOB: no log growth for ${STALL}s"; last_growth=$(date +%s)
  fi
  tick=$((tick+1))
  [ $((tick % 6)) -eq 0 ] && echo "HEARTBEAT $JOB: state=$state bytes=$size"
  sleep $INTERVAL
done
```

Arm with `Monitor(command: <script>, persistent: true, description: "job <id> (<experiment name>)")`. The `head -20` caps a signature avalanche; you'll read the real region yourself. Every terminal path emits a line — silence must never be a possible failure mode.

## Step 3 — Triage events (Tier 1, you)

**On `SIG`:** locate the match in the log (`grep -n` on betty), read a targeted window around it, and judge: real failure vs benign (a caught-and-retried error, "nan" inside a string, a warning). Benign → note it in the state file, continue.

**On `STALL`:** check `scontrol` state, GPU node health if visible, and whether the process is genuinely wedged (e.g. wandb shows metrics still flowing while stdout is quiet — some phases log rarely). Genuine hang → escalate.

**On `HEARTBEAT` (every ~30 min):** run the soft-anomaly health check. Pull recent metrics with one wandb call (or the last loss-bearing log lines if no wandb):

```bash
python -c "
import wandb; api = wandb.Api()
r = api.run('alelab/GREP-PRISM/<run_id>')
h = r.history(keys=['<loss_key>','<grad_norm_key>'], samples=100, pandas=False)
print(r.state); [print(row) for row in h[-15:]]
"
```

Escalate when any of these trips:
- loss is NaN/inf, or flat (< ~1% relative change) across the last two heartbeats during a phase where it previously moved
- grad norm pinned at 0, or exploding monotonically
- throughput (steps/s, tokens/s) sustained below ~50% of the run's earlier rate
- eval accuracy exactly 0 where comparable runs were nonzero
- wandb run state `crashed`/`failed` while SLURM still says RUNNING

**On `TERMINAL`:** sacct `COMPLETED` → Step 5. Anything else (`FAILED`, `OOM`, `TIMEOUT`, `NODE_FAIL`, `CANCELLED` not by the user) → escalate, with one exception:

**Infra-flake fast path (no Fable needed):** pure infrastructure deaths with healthy metrics — `NODE_FAIL`, preemption, an NCCL/network flake with no code implicated — you may resubmit directly (`ssh betty 'cd ~/sourcecode/GREP && sbatch scripts/<file>.sbatch'`), no code changes, PushNotification, increment the direct-resubmit count. Cap: 2 per lineage; a third infra death is a pattern, not a flake — escalate it.

## Step 4 — Escalation protocol (Tier 1 → Tier 2)

1. `PushNotification`: `"Escalating job <id>: <one-line reason>"`.
2. Build the handoff file `<scratchpad>/handoff-<jobid>-<n>.md`. The governing rule is **selection, not summarization**:
   - Copy log/code/config content **verbatim**. Never paraphrase, never "the traceback says roughly…".
   - Unsure whether a region matters → include it. Trim by dropping whole irrelevant regions, not by compressing what remains.
   - Target ≤ ~2,500 lines total.

   Sections, in this order:
   1. **JOB FACTS** — job id(s), sbatch path, config name + CLI overrides, node(s), submit/start times, log path on betty, wandb URL, escalation N of 3.
   2. **FAILURE SIGNAL** — verbatim: every traceback in full; the 100 log lines before the first error line; the final 50 lines of the log.
   3. **SBATCH SCRIPT** — full verbatim content.
   4. **CONFIG** — the Hydra config file(s) behind `--config-name`, verbatim (follow the defaults-list one level if small).
   5. **RECENT CHANGES** — local `git log --oneline -15` and `git status --short`; cluster `ssh betty 'cd ~/sourcecode/GREP && git log --oneline -5'` (exposes local/cluster skew); verbatim diff of any uncommitted changes touching files named in the traceback.
   6. **HEALTH TIMELINE** — for soft anomalies: the verbatim metric rows / log lines showing onset (when loss went flat, when throughput dropped).
   7. **PRIOR ESCALATIONS** — earlier Fable reports for this lineage, verbatim from the state file. This is what prevents fix-fail loops from retrying the same dead end.
   8. **MONITOR HYPOTHESIS** — one short paragraph, explicitly labeled an unverified hypothesis. Last, so it can't anchor the debugger's reading of the evidence.
3. Spawn the debugger **synchronously**:
   `Agent(subagent_type: "fable-debugger", model: "fable", run_in_background: false, prompt: "Handoff file: <absolute path>. Read it in full before touching anything else. Escalation <n> of 3 for this job lineage.")`
4. Append its report verbatim to the state file. `PushNotification` with the outcome: `"Job <id> fixed, resubmitted as <newid>: <root cause>"` or `"Fable declined on <id>: <reason>"`.
5. **NEW_JOB_ID returned:** TaskStop the old Monitor, run intake + arm for the new id. The state file (and its escalation count) carries over — the cap is 3 per *lineage*, not per SLURM id.
6. **NO_ACTION returned, or cap reached:** `PushNotification "Job <id> needs you: <reason>"`, stop monitoring that lineage, and end with the handoff + report paths and what you're unsure about.

## Step 5 — Completion

On `COMPLETED`: confirm it's a real success (sacct state, expected artifacts / final eval metrics in the log tail — a job that "completed" in 90 seconds did not train). `PushNotification "<jobid> completed: <headline metric>"`, TaskStop the monitor, and close with a short summary: final metrics, wandb link, checkpoint/output paths, anything odd worth a human look.

## Rules

- Push notifications only at: escalation start, escalation outcome, direct resubmit, completion, giving up. Never for heartbeats or benign signals.
- You never edit code and never fix-and-resubmit yourself — the only cluster mutation you're allowed is the infra-flake fast-path resubmit. Everything else goes through the fable-debugger.
- Multiple jobs → one Monitor and one state file each; triage events independently.
- If spawning the fable-debugger fails outright, PushNotification the user with the handoff path — the handoff file is exactly what they'd paste into a Fable session by hand.
