import gym
import numpy as np
from gym import spaces
from gym.envs.registration import register

class ToyEnv(gym.Env):
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None
    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Box(low=-np.array([1.]), high=np.array([1.]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.array([100., 100.]), high=np.array([100., 100.]), dtype=np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self._obs[0] += action * self._obs[1] / 10.0
        self._obs[1] += np.abs(action) * 0.05
        self._obs[1] = np.clip(self._obs[1], 0.0, 5.0)
        reward = 1.0 if np.abs(self._obs[0]) > 10.0 else 0.0
        done = True if reward > 0 else False
        return self._obs.copy(), reward, done, {}

    def reset(self, inital_pose=0.0):
        """Resets the state of the environment and returns an initial observation.

        Returns:
            observation (object): the initial observation.
        """
        # pos energy
        self._obs = np.array([inital_pose + np.random.rand() * 2 - 1.0, 0.0])
        return self._obs.copy()

    def render(self, mode='human'):
        return

    def close(self):
        return True

    def seed(self, seed=None):
        return

    def get_dataset(self):
        result = {}
        data = np.load("./dataset/toy-v0.npy", allow_pickle=True).item()
        result["observations"] = data["states"]
        result["next_observations"] = data["next_states"]
        result["actions"] = data["actions"]
        result["rewards"] = data["rewards"].squeeze()
        result["terminals"] = (data["done"]>0).squeeze()
        result["timeouts"] = ((data["done"]>0) &(~(data["is_finished"]>0))).squeeze()
        return result

class ToySingleEnv(ToyEnv):
    def get_dataset(self):
        result = {}
        data = np.load("./dataset/toy-singleside-v0.npy", allow_pickle=True).item()
        result["observations"] = data["states"]
        result["next_observations"] = data["next_states"]
        result["actions"] = data["actions"]
        result["rewards"] = data["rewards"].squeeze()
        result["terminals"] = (data["done"]>0).squeeze()
        result["timeouts"] = ((data["done"]>0) &(~(data["is_finished"]>0))).squeeze()
        return result

register(
    id='Toy-v0',
    entry_point='dataset.Bidirectional_Car_Env:ToyEnv',
    max_episode_steps=100,
    kwargs={
    }
)
register(
    id='toy-v0',
    entry_point='dataset.Bidirectional_Car_Env:ToyEnv',
    max_episode_steps=100,
    kwargs={
    }
)
register(
    id='Toy-singleside-v0',
    entry_point='dataset.Bidirectional_Car_Env:ToySingleEnv',
    max_episode_steps=100,
    kwargs={
    }
)
register(
    id='toy-singleside-v0',
    entry_point='dataset.Bidirectional_Car_Env:ToySingleEnv',
    max_episode_steps=100,
    kwargs={
    }
)

def generate_dataset():
    env = gym.make("Toy-v0")
    obs = env.reset()
    states = []
    next_states = []
    actions = []
    rewards =[]
    done = []
    is_finished = []
    
    choice = np.random.choice([0.95, -0.95])
    step_count = 0
    while True:
        a = np.random.rand() * 0.1 - 0.05 + choice
        obs_next, rew, d, info = env.step(a)
        step_count+=1
        states.append(obs)
        actions.append(a)
        rewards.append(rew)
        next_states.append(obs_next)
        done.append(d)
        is_finished.append(d if ("TimeLimit.truncated" not in info or info["TimeLimit.truncated"] == False) else False)
        if d:
            obs = env.reset(inital_pose=np.random.rand()*10 - 5.)
            choice = np.random.choice([0.95, -0.95])
            if step_count > 250000:
                break
        else:
            obs = obs_next
    
    step_count = 0
    while True:
        a = np.random.rand()*2.0-1.0
        obs_next, rew, d, info = env.step(a)
        step_count+=1
        states.append(obs)
        actions.append(a)
        rewards.append(rew)
        next_states.append(obs_next)
        done.append(d)
        is_finished.append(d if ("TimeLimit.truncated" not in info or info["TimeLimit.truncated"] == False) else False)
        if d:
            obs = env.reset(inital_pose=np.random.rand()*10 - 5.)
            if step_count > 250000:
                break
        else:
            obs = obs_next
            
    step_count = 0
    a = -1
    while True:
        if a < 0:
            if np.random.rand() < 0.1:
                a = np.random.rand() * 0.3 + 0.7
            else:
                a = -np.random.rand() * 0.3 - 0.7
        else:
            if np.random.rand() < 0.9:
                a = np.random.rand() * 0.3 + 0.7
            else:
                a = -np.random.rand() * 0.3 - 0.7
        obs_next, rew, d, info = env.step(a)
        step_count+=1
        states.append(obs)
        actions.append(a)
        rewards.append(rew)
        next_states.append(obs_next)
        done.append(d)
        is_finished.append(d if ("TimeLimit.truncated" not in info or info["TimeLimit.truncated"] == False) else False)
        if d:
            obs = env.reset(inital_pose=np.random.rand()*5. - 2.5)
            if step_count > 250000:
                break
        else:
            obs = obs_next

    step_count = 0
    a = 0.0
    while True:
        a = a + np.random.randn() * 0.5
        a = np.clip(a, -1, 1)
        obs_next, rew, d, info = env.step(a)
        step_count+=1
        states.append(obs)
        actions.append(a)
        rewards.append(rew)
        next_states.append(obs_next)
        done.append(d)
        is_finished.append(d if ("TimeLimit.truncated" not in info or info["TimeLimit.truncated"] == False) else False)
        if d:
            obs = env.reset(inital_pose=np.random.rand()*18. - 9.0)
            if step_count > 250000:
                break
        else:
            obs = obs_next
    
    states = np.stack(states)
    next_states = np.stack(next_states)
    actions = np.stack(actions)
    rewards = np.stack(rewards)
    done = np.stack(done)
    is_finished = np.stack(is_finished)
    assert np.max(rewards) == 1.0
    assert np.min(rewards) == 0.0
    print(done.shape[0], is_finished.shape[0])
    np.save("toy-v0.npy", {"states":states.astype(np.float32), "next_states": next_states.astype(np.float32), "actions":actions[:,None].astype(np.float32), "rewards":rewards[:,None].astype(np.float32), "done":done[:,None].astype(np.float32), "is_finished":is_finished[:,None].astype(np.float32)})

if __name__ == "__main__":
    generate_dataset()