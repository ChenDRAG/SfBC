# Offline Reinforcement Learning via High-Fidelity Generative Behavior Modeling

This is a pytorch implementation of SfBC: [Offline Reinforcement Learning via High-Fidelity Generative Behavior Modeling](https://arxiv.org/abs/2209.14548).

![Algorithm Overview](https://github.com/ChenDRAG/SfBC/blob/master/overview.PNG)

\* **For diffusion-based offline RL, we recommend trying our subsequent work, QGPO([paper](https://arxiv.org/abs/2304.12824); [Github](https://github.com/ChenDRAG/CEP-energy-guided-diffusion)). Compared with SfBC, QGPO has improved computational efficiency and noticeably better performance.**

## Requirements

- See conda requirements in `requirements.yml`

## Quick Start
Train the behavior model:

```shell
$ python3 train_behavior.py
```

Train the critic model and plot evaluation scores with tensorboard:

```shell
$ python3 train_critic.py
```

Evaluation only:

```shell
$ python3 evaluation.py
```

## Citing
If you find this code release useful, please reference in your paper:
```
@inproceedings{
chen2023offline,
title={Offline Reinforcement Learning via High-Fidelity Generative Behavior Modeling},
author={Huayu Chen and Cheng Lu and Chengyang Ying and Hang Su and Jun Zhu},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
}
```

## Note
+ Contact us at: chenhuay17@gmail.com if you have any question.
