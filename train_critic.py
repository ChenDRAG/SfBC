import os
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
targe_Q_mode = False

def train_critic(args, score_model, data_loader, optimizer, start_epoch=0):
    if args.reset_critic:
        import copy
        bk_model_sd = copy.deepcopy(score_model.q[0].state_dict())
    n_epochs = args.K * 100 - 1
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x, condition in data_loader:
            x = x.to(args.device)
            condition = condition.to(args.device)
            loss = torch.mean((score_model.calculateQ(condition, x[:,1:]) - x[:,:1])**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            score_model.condition = None
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        if (epoch % 5 == 4) or epoch==0:
            torch.save(score_model.q[0].state_dict(), os.path.join("./models", str(args.expid), "critic_ckpt{}.pth".format(epoch+1)))
            # evaluation
            if args.evaluate_while_training_critic:
                envs = args.eval_func(score_model.select_actions)
                mean = np.mean([envs[i].dbag_return for i in range(args.seed_per_evaluation)])
                std = np.std([envs[i].dbag_return for i in range(args.seed_per_evaluation)])
                args.writer.add_scalar("eval/rew", mean, global_step=epoch)
                args.writer.add_scalar("eval/std", std, global_step=epoch)
        if epoch in [99, 199, 299, 399]:
            data_loader.dataset.update_returns(score_model)
            if args.reset_critic:
                score_model.q[0].load_state_dict(bk_model_sd) 
                bk_model_sd = copy.deepcopy(score_model.q[0].state_dict())
                optimizer = Adam(score_model.q[0].parameters(), lr=1e-3)
        if args.writer:
            args.writer.add_scalar("critic/loss", avg_loss / num_items, global_step=epoch)

def critic(args):
    for dir in ["./models", "./logs"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./models", str(args.expid))):
        os.makedirs(os.path.join("./models", str(args.expid)))
    writer = SummaryWriter("./logs/" + str(args.expid))
    
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.eval_func = functools.partial(pallaral_eval_policy, env_name=args.env, seed=args.seed, eval_episodes=args.seed_per_evaluation, track_obs=False, select_per_state=args.select_per_state, diffusion_steps=args.diffusion_steps)
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
    # if args.actor_loadpath is not specifided, should be determined by expid and args.actor_load_epoch
    if args.actor_load_setting is None:
        args.actor_loadpath = os.path.join("./models", str(args.expid), "ckpt{}.pth".format(args.actor_load_epoch))
    else:
        args.actor_loadpath = os.path.join("./models", args.env + str(args.seed) + args.actor_load_setting, "ckpt{}.pth".format(args.actor_load_epoch))
    print("loading actor...")
    ckpt = torch.load(args.actor_loadpath, map_location=args.device)
    score_model.load_state_dict(ckpt)
    
    dataset = Diffusion_buffer(args)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # if fake action cannot be find, we should fake action in the actor load path
    if os.path.exists(args.actor_loadpath+ "actions{}.npy".format(args.diffusion_steps)):
        dataset.fake_actions = np.load(args.actor_loadpath+ "actions{}.npy".format(args.diffusion_steps))
    else:
        allstates = dataset.states[:]
        all_resuls = []
        for states in tqdm.tqdm(np.array_split(allstates, allstates.shape[0] // 256 + 1)):
            all_resuls.append(score_model.sample(states, sample_per_state=16, diffusion_steps=args.diffusion_steps))
        returns = np.concatenate([res[0] for res in all_resuls]) # <bz, 16, 1>    
        actions = np.concatenate([res[1] for res in all_resuls])
        dataset.fake_actions = actions
        np.save(args.actor_loadpath+ "actions{}.npy".format(args.diffusion_steps), actions)
    
    print("training critic")
    optimizer = Adam(score_model.q[0].parameters(), lr=1e-3)
    # train_critic(args, score_model, data_loader, optimizer, start_epoch=250)
    train_critic(args, score_model, data_loader, optimizer, start_epoch=0)
    print("finished")

if __name__ == "__main__":
    args = get_args()
    if "antmaze" not in args.env:
        args.sample_per_epoch=1000000
    else:
        args.sample_per_epoch=2000000
    critic(args)