---
name: git-state
description: Summarizes the current git staging state
---

# Git state summarizer

## Instructions
Run the following single bash command to gather all needed info at once:

```bash
git status && echo "---" && git rev-parse --short HEAD && echo "---" && git log --oneline --graph --decorate | head -4 && echo "---" && git diff --cached --stat && echo "---" && git diff --stat
```

Add a "SUGGESTED  COMMIT MESSAGE" section to the output. Suggest a commit message for the staged changes.

Then offer the user a summary of:

1. The current state of the staging area.
2. The meaning of the changes in the context of the repo.
3. Whether the changes seem to be ready to commit.

The output should be a concise summary intended to aid the user in deciding the next steps in the project.

## Output format
- Keep the entire output to a single laptop screen (under ~30 lines total)
- Do NOT use markdown tables or `###` headers — use `**BOLD**` for section titles
- Use short indented lines inside code blocks, not prose paragraphs
- Start with: `**Branch: `<name>`** — <N> commit(s) ahead/behind remote, unpushed/pushed  `<hash>``
- Include the first 4 lines of the log in a code block
- Section titles are `**UPPERCASE (+N/-M)**` bold, with totals where applicable
- For STAGED: one line per file — `M  path/to/file  +N  brief annotation of what/why`
- For UNSTAGED: one line per file — `M  path/to/file  brief annotation`
- For UNTRACKED: list inline on one line
- For COMMIT READINESS: `[+]` good signals, `[!]` warnings/blockers — be specific to the repo context
- For notebooks, don't read diffs as they can be too large and a waste of tokens.

## Example output

**Branch: `e2_rpearl_improvements`** — 1 commit ahead of remote, unpushed  `4470798`

```
* 4470798 (HEAD -> e2_rpearl_improvements) WIP: Major refctors for graph-based planner…
*   24bd5d9 (origin/main, origin/e2_rpearl_improvements) Merge pull request #1 from …
|\
| * a2ba3b7 (tag: e1, origin/e1_distillation_baseline_quakerlives) e1 refactors and results
```

**STAGED (+109/-5)**
```
M  src/prism/eval/callbacks.py          +86  new GradientDebugCallback (grad norms, PE health → W&B)
M  src/prism/models/gnn_llm.py           +5  pe_proj: Linear → Sequential(Linear, LayerNorm)
M  src/prism/training/train_v2.py        +7  wire up GradientDebugCallback; switch eval data path
M  notebooks/dev_inference_check.ipynb  +13  kernel crash artifact — consider unstaging
```

**UNSTAGED**
```
M  AGENTS.md                        minor doc edit (anti-defensive-programming note)
M  e1_distillation_baseline_results.ipynb  large output churn (~3k lines), likely re-execution
```

**UNTRACKED (ignored for commit)**
```
.claude/  .cursor/  CLAUDE.md  SPINE/  checkpoints/
```

**COMMIT READINESS: mostly ready**
```
[+] Core changes coherent: RPEARL debug observability + LayerNorm stabilization
[!] Unstage the notebook — kernel crash output is noise
[!] HEAD commit is "WIP" — confirm eval infinite loop is resolved before pushing
[!] Stage AGENTS.md if it belongs with this commit
```

## IMPORTANT RESTRICTIONS
When using this skill, you are NOT ALLOWED to: make changes to the code,
change the git state (you can look into other branches without checking out or stashing),
or making any edits.
