#!/bin/bash
CONDA_RUN="conda run -n GREP-PRISM --no-capture-output"

tmux new-session -d -s rpearl "cd $(pwd) && $CONDA_RUN env CUDA_VISIBLE_DEVICES=0 python -m prism.training.train_v2 experiments/e2_rpearl_llm.yaml; echo 'rpearl DONE (exit $?)'; read"
echo "Started rpearl on GPU0 (tmux session: rpearl)"

tmux new-session -d -s llm "cd $(pwd) && $CONDA_RUN env CUDA_VISIBLE_DEVICES=1 python -m prism.training.train_v2 experiments/e2_llm.yaml; echo 'llm DONE (exit $?)'; read"
echo "Started llm on GPU1 (tmux session: llm)"

echo "Attach with: tmux attach -t rpearl  OR  tmux attach -t llm"

