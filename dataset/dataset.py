import torch
import torch.nn as nn
import gym
import d4rl
import numpy as np
import functools
import copy
import os
import torch.nn.functional as F
import tqdm
from scipy.special import softmax
MAX_BZ_SIZE = 1024
soft_Q_update = True

class Diffusion_buffer(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args=args
        self.normalise_return = args.normalise_return
        data = self._load_data(args)
        self.actions = data["actions"]
        self.states = data["states"]
        self.rewards = data["rewards"]
        self.done = data["done"]
        self.is_finished = data["is_finished"]
        
        returns = data["returns"]
        self.raw_returns = [returns]
        self.raw_values = []
        self.returns_mean = np.mean(returns)
        self.returns_std = np.maximum(np.std(returns), 0.1)
        print("returns mean {}  std {}".format(self.returns_mean, self.returns_std))
        if self.normalise_return:
            returns = (returns - self.returns_mean) / self.returns_std
            print("returns normalised at mean {}, std {}".format(self.returns_mean, self.returns_std))
            self.args.returns_mean = self.returns_mean
            self.args.returns_std = self.returns_std
        else:
            print("no normal")
        self.ys = np.concatenate([returns, self.actions], axis=-1)
        self.ys = self.ys.astype(np.float32)

        self.len = self.states.shape[0]
        # make sure same number of data points exist in all tasks
        self.fake_len = int(np.maximum(np.round(args.sample_per_epoch / self.len), 1)) * self.len
        print(self.len, "data loaded", self.fake_len, "data faked")
    
        
    def __getitem__(self, index):
        data = self.ys[index % self.len]
        condition = self.states[index % self.len]
        return data, condition

    def __add__(self, other):
        pass
    def __len__(self):
        return self.fake_len
    
    def _load_data(self, args):
        if "hopper" in args.env or "walker" in args.env or "halfcheetah" in args.env or "pen" in args.env or "hammer" in args.env or "door" in args.env or "relocate" in args.env or "kitchen" in args.env:
            t = gym.make(args.env)
            dataset = t.get_dataset()
            if not dataset["terminals"][-1]:
                dataset["timeouts"][-1] = True
            data = {}
            assert not np.any(dataset["terminals"] & dataset["timeouts"])
            data["states"] = dataset["observations"]
            data["actions"] = dataset["actions"]
            # data["next_states"] = dataset["next_observations"]
            data["rewards"] = dataset["rewards"][:, None]
            data["done"] = (dataset["terminals"] | dataset["timeouts"])[:, None]
            data["is_finished"] = dataset["terminals"][:, None]
            assert data["done"][-1, 0]
            data["returns"] = np.zeros((data["states"].shape[0], 1))
            last = 0
            for i in range(data["returns"].shape[0] - 1, -1, -1):
                last = data["rewards"][i, 0] + 0.99 * last * (1. - data["done"][i, 0])
                data["returns"][i, 0] = last
        elif "maze" in args.env:
            t = gym.make(args.env)
            dataset = t.get_dataset()
            if dataset["terminals"][-1] == False:
                dataset["timeouts"][-1] = True
            assert "observations_next" not in dataset and "next_observations" not in dataset 
            assert dataset["timeouts"].shape[0] == dataset["observations"].shape[0]
            data = {}
            if "maze2d" in args.env:
                assert np.sum(dataset["terminals"]) == 0
                assert np.max(dataset["observations"][1:,0] - dataset["observations"][:-1,0]) < 1.0
                data["states"] = dataset["observations"][:-1, :]
                data["next_states"] = dataset["observations"][1:, :] 
                data["done"] = np.zeros((data["states"].shape[0], 1))
                # TODO check
                data["done"][-1,0]  = 1
                data["is_finished"] = np.zeros((data["states"].shape[0], 1))
                data["actions"] = dataset["actions"][:-1, :]
                data["rewards"] = dataset["rewards"][:-1, None]
                data["returns"] = np.zeros((data["states"].shape[0], 1))
                last = 0
                for i in range(data["returns"].shape[0] - 1, -1, -1):
                    last = data["rewards"][i, 0] + 0.99 * last * (1. - data["done"][i, 0])
                    data["returns"][i, 0] = last
            elif "antmaze" in args.env:
                if args.env == "antmaze-medium-play-v2":
                    # solve the little bug within this dataset
                    rewid = np.where(dataset["rewards"]>0.001)[0]
                    positions = dataset["observations"][rewid, :2]
                    badid = rewid[~np.all(positions > 19.0, axis=1)]
                    print("{} badid detected".format(badid.shape[0]))
                    dataset["rewards"][badid] = 0.0
                    dataset["terminals"][badid] = 0
                assert set(np.where(np.abs(dataset["observations"][1:,0] - dataset["observations"][:-1,0]) > 1.)[0]).issubset(set(np.where(dataset["timeouts"])[0]))
                assert np.all(np.where(dataset["rewards"])[0] == np.where(dataset["terminals"])[0])
                doneid = dataset["terminals"] | dataset["timeouts"]
                start_id = np.where(doneid)[0]+1
                assert start_id[-1] == doneid.shape[0]
                assert start_id[0] != 0
                start_id = [0] + [i for i in start_id]
                data = {"states":[], "next_states":[], "done":[], "is_finished":[], "rewards":[], "actions":[]}
                for i in range(len(start_id) - 1):
                    if start_id[i+1] - start_id[i] < 5:
                        continue
                    if dataset["terminals"][start_id[i+1]-1]:
                        data["states"].append(dataset["observations"][start_id[i]: start_id[i+1]])
                        next_states = np.zeros_like(data["states"][-1])
                        next_states[:-1] = data["states"][-1][1:]
                        data["next_states"].append(next_states)
                        data["actions"].append(dataset["actions"][start_id[i]: start_id[i+1]])
                        data["rewards"].append(dataset["rewards"][start_id[i]: start_id[i+1], None])
                        done = np.zeros((data["states"][-1].shape[0], 1))
                        done[-1, 0] = 1
                        data["done"].append(done)
                        data["is_finished"].append(done)
                    elif dataset["timeouts"][start_id[i+1]-1]:
                        data["states"].append(dataset["observations"][start_id[i]: start_id[i+1]-1])
                        data["next_states"].append(dataset["observations"][start_id[i]+1: start_id[i+1]])
                        data["actions"].append(dataset["actions"][start_id[i]: start_id[i+1]-1])
                        data["rewards"].append(dataset["rewards"][start_id[i]: start_id[i+1]-1, None])
                        done = np.zeros((data["states"][-1].shape[0], 1))
                        done[-1, 0] = 1
                        data["done"].append(done)
                        data["is_finished"].append(np.zeros_like(data["rewards"][-1]))
                    else:
                        assert False
                for k in ["states", "next_states", "done", "is_finished", "rewards", "actions"]:
                    data[k] = np.concatenate(data[k])
                    size = data[k].shape[0]
                print("data num {}".format(size))
                for k in ["states", "next_states", "done", "is_finished", "rewards", "actions"]:
                    assert data[k].shape[0] == size
                    assert data[k].ndim == 2
                    # bootstrap by 0 ignore is_finished
                data["returns"] = np.zeros((data["states"].shape[0], 1))
                last = 0
                for i in range(data["returns"].shape[0] - 1, -1, -1):
                    last = data["rewards"][i, 0] + 0.99 * last * (1. - data["done"][i, 0])
                    data["returns"][i, 0] = last
        elif "Toy" in args.env or "toy" in args.env :
            if "single" in args.env:
                data = np.load("./dataset/toy-singleside-v0.npy", allow_pickle=True).item()
            else:
                data = np.load("./dataset/toy-v0.npy", allow_pickle=True).item()
            data["returns"] = np.zeros((data["states"].shape[0], 1))
            last = 0
            for i in range(data["returns"].shape[0] - 1, -1, -1):
                last = data["rewards"][i, 0] + 0.99 * last * (1. - data["done"][i, 0])
                data["returns"][i, 0] = last
        else:
            raise NotImplementedError
        return data
    
    def update_returns(self, score_model):
        assert self.states.shape[0] == self.fake_actions.shape[0]
        qs = []
        for states, actions in tqdm.tqdm(zip(np.array_split(self.states, self.states.shape[0] // 128 + 1), np.array_split(self.fake_actions, self.states.shape[0] // 128 + 1))):
            with torch.no_grad():
                states = torch.FloatTensor(states).to("cuda")
                actions = torch.FloatTensor(actions).to("cuda")
                states = torch.repeat_interleave(states, actions.shape[1], dim=0)
                q = score_model.calculateQ(states, actions.reshape((states.shape[0], actions.shape[-1])))
                q = q.reshape((actions.shape[0], actions.shape[1]))
                qs.append(q.cpu().numpy())
        values = np.concatenate(qs)

        self.raw_values.append(values)
        if soft_Q_update:
            values = np.sum(softmax(self.args.alpha * values, axis=-1) * values, axis=-1, keepdims=1)
        else:
            values = np.percentile(values, 85, axis=-1, keepdims=1)
        if self.normalise_return:
            values = values * self.returns_std + self.returns_mean
        assert values.ndim == 2
        assert values.shape[0] == self.states.shape[0]
        returns = np.zeros_like(values)
        last = 0
        num_truncated_traj = 0
        for i in range(returns.shape[0] - 1, -1, -1):
            bootstrap = self.rewards[i, 0] + 0.99 * last * (1.0 - self.done[i, 0])
            imagainary = values[i, 0]
            if bootstrap > imagainary:
                returns[i, 0] = bootstrap
            else:
                returns[i, 0] = imagainary
                num_truncated_traj += 1
            last = returns[i, 0]
        print("num_truncated_traj perc", num_truncated_traj / returns.shape[0])
        self.raw_returns.append(returns)
        self.returns_mean = np.mean(returns)
        self.returns_std = np.maximum(np.std(returns), 0.1)
        print("returns mean {}  std {}".format(self.returns_mean, self.returns_std))
        if self.normalise_return:
            returns = (returns - self.returns_mean) / self.returns_std
            print("returns normalised at mean {}, std {}".format(self.returns_mean, self.returns_std))
            self.args.returns_mean = self.returns_mean
            self.args.returns_std = self.returns_std
        else:
            print("no normal")

        self.ys = np.concatenate([returns, self.actions], axis=-1)

        self.ys = self.ys.astype(np.float32)
        print("update returns finished")