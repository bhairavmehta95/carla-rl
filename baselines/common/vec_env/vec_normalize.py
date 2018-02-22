from baselines.common.vec_env import VecEnvWrapper
from baselines.common.running_mean_std import RunningMeanStd
import numpy as np

class VecNormalize(VecEnvWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.spaces[0].shape) if ob else None

        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs_tuple, rews, news, infos = self.venv.step_wait()
        obs_img, obs_measure = self.process_obs(obs_tuple)
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs_img)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        return obs, obs_measure, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        """
        Reset all environments
        """
        obs_tuple = self.venv.reset()
        obs_img, obs_measure = self.process_obs(obs_tuple)
        return self._obfilt(obs_img), obs_measure

    def process_obs(self, obs_tuple):
        obs_tuple = np.array(obs_tuple)
        obs_img = []
        obs_measure = []
        
        for i in range(obs_tuple.shape[0]):
            obs_img.append(obs_tuple[i][0])
            obs_measure.append(obs_tuple[i][1])

        return np.array(obs_img), np.array(obs_measure)    
