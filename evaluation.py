import os
import re
import gym
import d4rl
import scipy
import tqdm
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import Bidirectional_Car_Env
from diffusion_SDE.loss import loss_fn
from diffusion_SDE.schedule import marginal_prob_std
from diffusion_SDE.model import ScoreNet, MlpScoreNet
from utils import get_args, pallaral_eval_policy
from dataset.dataset import Diffusion_buffer

def eval(args):
    for dir in ["./models", "./eval_logs", "./results"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./models", str(args.expid))):
        os.makedirs(os.path.join("./models", str(args.expid)))
    if not os.path.exists(os.path.join("./results", str(args.expid))):
        os.makedirs(os.path.join("./results", str(args.expid)))
    writer = SummaryWriter("./eval_logs/" + str(args.expid))
    
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.eval_func = functools.partial(pallaral_eval_policy, env_name=args.env, seed=args.seed, eval_episodes=20, track_obs=False, select_per_state=args.select_per_state, diffusion_steps=args.diffusion_steps)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    args.writer = writer
    
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=args.sigma, device=args.device)
    args.marginal_prob_std_fn = marginal_prob_std_fn
    if args.actor_type == "large":
        score_model= ScoreNet(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
    elif args.actor_type == "small":
        score_model= MlpScoreNet(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
    score_model.q[0].to(args.device)

    
    actor_loadpath = os.path.join("./models", args.env + str(args.seed) + args.actor_load_setting, "ckpt{}.pth".format(args.actor_load_epoch))
    print("loading actor...")
    ckpt = torch.load(actor_loadpath, map_location=args.device)
    score_model.load_state_dict(ckpt)

    critic_files = [f for f in os.listdir(os.path.join("./models", args.env + str(args.seed) + args.critic_load_setting)) if "critic_ckpt" in f]
    critic_load_epochs = sorted([int(re.findall("critic_ckpt(.*?)pth",f)[0][:-1]) for f in critic_files])
    print(critic_load_epochs)
    for ct in critic_load_epochs:
        critic_loadpath = os.path.join("./models", args.env + str(args.seed) + args.critic_load_setting, "critic_ckpt{}.pth".format(str(ct)))
        print("loading critic {}...".format(critic_loadpath))
        ckpt = torch.load(critic_loadpath, map_location=args.device)
        score_model.q[0].load_state_dict(ckpt)

        # evaluation
        envs = args.eval_func(score_model.select_actions)
        mean = np.mean([envs[i].dbag_return for i in range(10)])
        std = np.std([envs[i].dbag_return for i in range(10)])
        args.writer.add_scalar("eval/rew", mean, global_step=ct-1)
        args.writer.add_scalar("eval/std", std, global_step=ct-1)

    print("finished")

if __name__ == "__main__":
    args = get_args()
    eval(args)