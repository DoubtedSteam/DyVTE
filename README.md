# DyVTE: Dynamic Visual Token Exiting

## Install

1. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

2. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

If met any problem, please refer to the installtion of LLaVA.

## Training

1. Init weight
```Shell
python init_weight.py
```

2. Start Training
```Shell
bash scripts/v1_5/finetune_visual_gate.sh
```

## Evaluation

Please refer to the evaluation of LLaVA.

For example, to test on the gqa dataset:

```Shell
bash scripts/v1_5/eval/gqa.sh
```