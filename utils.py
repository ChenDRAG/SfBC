import os
import argparse
import gym
import d4rl
import numpy as np
import time
from tensorboard.backend.event_processing import event_accumulator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="halfcheetah-medium-v2") # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--expid", default="default", type=str)    # 
    parser.add_argument("--device", default="cuda", type=str)      #
    parser.add_argument("--save_model", default=1, type=int)       #
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--sigma', type=float, default=40.0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--alpha', type=float, default=20.0)
    parser.add_argument('--train', type=int, default=0)
    parser.add_argument('--n_behavior_epochs', type=int, default=600)
    parser.add_argument('--normalise_return', type=int, default=1)
    parser.add_argument('--critic_type', type=str, default=None)
    parser.add_argument('--actor_type', type=str, default="large")
    parser.add_argument('--actor_load_epoch', type=int, default=600)
    parser.add_argument('--actor_load_setting', type=str, default=None)
    parser.add_argument('--critic_load_setting', type=str, default=None)
    parser.add_argument('--diffusion_steps', type=int, default=15)
    parser.add_argument('--sample_per_epoch', type=int, default=4000000)
    parser.add_argument('--reset_critic', type=int, default=1)
    parser.add_argument('--seed_per_evaluation', type=int, default=10)
    parser.add_argument('--evaluate_while_training_critic', type=int, default=1)
    parser.add_argument('--K', type=int, default=2)
    print("**************************")
    args = parser.parse_known_args()[0]
    args.debug = 0
    if args.debug:
        args.actor_epoch =1
        args.critic_epoch =1
        args.env = "antmaze-medium-play-v2"
    if args.critic_type is None:
        args.critic_type = "large" if "antmaze-large" in args.env else "small"
    if not ("halfcheetah" in args.env or "hopper" in args.env or "walker" in args.env):
        args.select_per_state = 1
    else:
        args.select_per_state = 4 # stablize performance
    print(args)
    return args

def pallaral_eval_policy(policy_fn, env_name, seed, eval_episodes=20, track_obs=False, select_per_state=1, diffusion_steps=15):
    del track_obs
    eval_envs = []
    for i in range(eval_episodes):
        env = gym.make(env_name)
        eval_envs.append(env)
        env.seed(seed + 1001 + i)
        env.dbag_state = env.reset()
        env.dbag_return = 0.0
        env.alpha = 100 # 100 could be considered as deterministic sampling since it's now extremely sensitive to normalized Q(s, a)
        env.select_per_state = select_per_state
    ori_eval_envs = [env for env in eval_envs]
    import time
    t = time.time()
    while len(eval_envs) > 0:
        new_eval_envs = []
        states = np.stack([env.dbag_state for env in eval_envs])
        actions = policy_fn(states, sample_per_state=32, select_per_state=[env.select_per_state for env in eval_envs], alpha=[env.alpha for env in eval_envs], replace=False, weighted_mean=False, diffusion_steps=diffusion_steps)
        for i, env in enumerate(eval_envs):
            state, reward, done, info = env.step(actions[i])
            env.dbag_return += reward
            env.dbag_state = state
            if not done:
                new_eval_envs.append(env)
        eval_envs = new_eval_envs
    print(time.time() - t)
    t = time.time()
   
    return ori_eval_envs


def plot_tools(folder_name, setting_name, task, seed=0, plt=None):
    if isinstance(seed, list):
        ys = []
        stds = []
        for s in seed:
            ts, y, std = plot_tools(folder_name, setting_name, task, s, None)
            ys.append(y)
            stds.append(std)
        ys = np.stack(ys)
        stds = np.std(ys, axis=0)
        ys = np.mean(ys, axis=0)
        if plt:
            plt.plot(ts, ys)
            plt.fill_between(ts, ys-stds, ys+stds, alpha=0.4)
        return ts, ys, stds
    else:
        tfevent_file = os.path.join(folder_name, task+str(seed)+setting_name)
        env = gym.make(task)
        ea = event_accumulator.EventAccumulator(tfevent_file)
        ea.Reload()
        ts = []
        ys = []
        stds = []
        for test_reward in ea.scalars.Items('eval/rew'):
            ts.append(test_reward.step)
            ys.append(env.get_normalized_score(test_reward.value))
        for test_reward in ea.scalars.Items('eval/std'):
            stds.append(env.get_normalized_score(test_reward.value))
        ts = np.array(ts)
        ys = np.array(ys) * 100
        stds = np.array(stds) * 100
        
        # deal with a special condition
        firstid = np.where(ts==0)[0][-1]
        ts = ts[firstid:]
        ys = ys[firstid:]
        stds = stds[firstid:]

        if plt:
            plt.plot(ts, ys)
            try:
                plt.fill_between(ts, ys-stds, ys+stds, alpha=0.4)
            except:
                print(tfevent_file + "  bad file")
                ys = ys[:100]
                stds = stds[:100]
                ts = ts[:100]
        return ts, ys, stds
