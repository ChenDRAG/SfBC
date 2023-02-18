# Offline Reinforcement Learning via High-Fidelity Generative Behavior Modeling

This is a pytorch implementation of SfBC: [Offline Reinforcement Learning via High-Fidelity Generative Behavior Modeling](https://arxiv.org/abs/2209.14548).

![Algorithm Overview](https://github.com/ChenDRAG/SfBC/blob/master/overview.PNG)

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
@article{chen2022offline,
  title={Offline Reinforcement Learning via High-Fidelity Generative Behavior Modeling},
  author={Chen, Huayu and Lu, Cheng and Ying, Chengyang and Su, Hang and Zhu, Jun},
  journal={arXiv preprint arXiv:2209.14548},
  year={2022}
}
```

## Note
+ Contact us at: chenhuay17@gmail.com if you have any question.
