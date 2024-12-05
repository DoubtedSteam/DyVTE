# **Accelerating Multimodal Large Language Models via Dynamic Visual-Token Exit and the Empirical Findings**
> [Qiong Wu](https://scholar.google.com/citations?hl=en&user=HyKLYKYAAAAJ)<sup>1</sup>,  Wenhao Lin<sup>1</sup>, Weihao Ye<sup>1</sup>, Yiyi Zhou<sup>1</sup>, [Xiaoshuai Sun](https://sites.google.com/view/xssun)<sup>1</sup>, [Rongrong Ji](https://mac.xmu.edu.cn/rrji/)<sup>1</sup>

> <sup>1</sup>Media Analytics and Computing Lab, Department of Artificial Intelligence, School of Informatics, Xiamen University  

## Abstract

The excessive use of visual tokens in existing *Multimoal Large Language Models* (MLLMs) often exhibits obvious redundancy and brings in prohibitively expensive computation. 
To gain insights into this problem, we first conduct extensive empirical studies on the attention behaviors of MLLMs, and summarize three main inference stages in MLLMs: 
*(i)* *Early fusion* between tokens is first accomplished quickly. 
*(ii)* *Intra-modality modeling* then comes to play. 
*(iii)* *Multimodal reasoning* resumes and lasts until the end of inference. 
In particular, we reveal that visual tokens will stop contributing to reasoning when the text tokens receive enough image information, yielding obvious visual redundancy. 
Based on these generalized observations, we propose a simple yet effective method to improve the efficiency of MLLMs, termed *dynamic visual-token exit* (DyVTE). 
DyVTE uses lightweight hyper-networks to perceive the text token status and decide the removal of all visual tokens after a certain layer, thereby addressing the observed visual redundancy.
To validate VTE, we apply it to a set of MLLMs, including LLaVA, VILA, Eagle and InternVL, and conduct extensive experiments on a bunch of benchmarks.
The experiment results not only show the effectiveness of our VTE in improving MLLMs' efficiency, but also yield the  general modeling patterns of MLLMs, well facilitating the in-depth understanding of MLLMs. 

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
