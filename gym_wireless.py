import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os,sys
from gymnasium import spaces
import string
from stable_baselines3.common.env_checker import check_env
import torch as T
import torch.nn as nn
from gymnasium import spaces
from model import LLM
from stable_baselines3 import PPO,SAC,DDPG
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor




class Transmitter:
    '''
    Class for base station with various parameters
    '''
    def __init__(self,loc,fc,Pt,Ga):
        '''
        :param loc: the location of the base station
        :param fc: the carrier frequency that the base station uses
        :param Pt: the transmit power of the base station
        :param Ga: the antenna gain
        '''
        self.loc = loc
        self.height = loc[-1]
        self.fc = fc
        self.transmit_power = Pt
        self.antenna_gain = Ga


    def get_eirp(self):
        """
        Returns the Effective Isotropic Radiated Power (EIRP)
        :return: EIRP
        """
        return self.transmit_power + self.antenna_gain

class Receiver:
    """
    Class for receiver points
    """
    def __init__(self,loc):
        """
        :param loc: the location where the measurement is taken
        """
        self.loc = loc


class LOS_Env(gym.Env):

    def __init__(self,N,mean):
        super(LOS_Env, self).__init__()
        self.n_receivers = N
        self.rec_mean = mean


        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
        # self.observation_space = spaces.Text(
        #     max_length=512,charset=string.printable
        # )

        self.observation_space = spaces.Box(low=-500, high=500, shape=(N,3), dtype=np.float32)

    def initialize_receivers(self):

        self.receivers = []
        for i in range(self.n_receivers):
            # rc_loc = np.random.uniform(3)-0.5) + self.rec_mean  # randomly choosing a location for transmitter
            # rc_loc[-1] = 0  # the height of the transmitter will be fixed for now
            rc_loc = np.random.uniform(low=-250+self.rec_mean, high=250+self.rec_mean, size=(3,))
            rc_loc[-1] = 0
            self.receivers.append(Receiver(rc_loc))

    def get_receiver_loc(self):
        locations = np.zeros((self.n_receivers,3))
        for i in range(self.n_receivers):
            locations[i,:] = self.receivers[i].loc
        return locations
    def initialize_transmitter(self,loc,fc=6e9,Pt=70,Ga=50):
        self.transmitter = Transmitter(loc,fc,Pt,Ga)
    def get_distance(self):
        self.distances = np.zeros((self.n_receivers,))
        for i in range(self.n_receivers):
            self.distances[i] = np.linalg.norm(self.transmitter.loc - self.receivers[i].loc)
            # self.distances[i] = np.linalg.norm(self.transmitter.loc - self.receivers[i].loc)
        self.distances
    def free_space_path_loss(self):
        """
        Computes the path loss in db for N points
        :return: the path loss for N points
        """
        c = 3e8
        lamda_c = c / self.transmitter.fc
        # return 10 * np.log10((4 * np.pi * self.distances / lamda_c) ** 2)
        fspl =10 * np.log10((4 * np.pi * self.distances / lamda_c) ** 2)
        return fspl

    def get_rssi(self):
        self.get_distance()
        path_loss = self.free_space_path_loss()
        eirp = self.transmitter.get_eirp()
        return eirp - path_loss

    def get_prompt(self):
        locations = self.get_receiver_loc()
        prompt = ""
        for i,loc in enumerate(locations):
            prompt += f"Location {i+1}: ({locations[i,0]:.2f},{locations[i,1]:.2f},{locations[i,2]:.2f}), "
        return prompt

    def reset(self, seed=None, options=None):

        super().reset(seed=seed, options=options)
        tr_loc = np.random.uniform(low=-500, high=500, size=(3,))  # randomly choosing a location for transmitter
        tr_loc[-1] = 50  # the height of the transmitter will be fixed for now
        self.initialize_transmitter(tr_loc, 6e9, 78.5,
                                    47.5)  # initialize one transmitter later should convert this to a list
        self.initialize_receivers()
        locations = self.get_receiver_loc()
        return locations.astype(np.float32), {}  # empty info dict

    def step(self, action):
        action = np.append(500*action,np.array([50])) #the action is expected to be (x,x,50)
        self.initialize_transmitter(action)  # initialize one transmitter later should convert this to a list
        rssi = self.get_rssi()


        reward = np.mean(rssi)
        locations = self.get_receiver_loc()
        terminated = False
        truncated = False
        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (
            locations.astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        rssi = self.get_rssi()
        fig = plt.figure()
        locs = self.get_receiver_loc()
        mean_loc = np.mean(locs, axis=0)
        for i in range(self.n_receivers):
            plt.scatter(self.receivers[i].loc[0], self.receivers[i].loc[1], marker="o")
            plt.annotate("{:.2f}".format(rssi[i]), self.receivers[i].loc[:2], (
                self.receivers[i].loc[0] - 20, self.receivers[i].loc[1] + 15), fontsize=6)
        plt.scatter(self.transmitter.loc[0],
                    self.transmitter.loc[1], marker="D", s=200)
        reward = np.mean(self.get_rssi())
        plt.annotate("{:.2f}".format(reward), self.transmitter.loc[:2],
                     (self.transmitter.loc[0] - 20, self.transmitter.loc[1]),
                     fontsize=10)
        self.initialize_transmitter(mean_loc)
        plt.scatter(self.transmitter.loc[0],
                    self.transmitter.loc[1], marker="v", s=200, color='r')
        reward = np.mean(self.get_rssi())
        plt.annotate("{:.2f}".format(reward), self.transmitter.loc[:2],
                     (self.transmitter.loc[0] - 20, self.transmitter.loc[1]),
                     fontsize=10)
        plt.xlim([-500, 500])
        plt.ylim([-500, 500])
        plt.show()
        plt.close()

    def close(self):
        pass


class LLM_FE(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, model_name= 'distilbert-base-uncased'):
        super().__init__(observation_space, features_dim)

        self.llm = LLM(model_name,features_dim)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, observations: T.Tensor) -> T.Tensor:
        locations = observations.cpu().squeeze().numpy()
        prompt = ""
        for i, loc in enumerate(locations):
            prompt += f"Location {i + 1}: ({locations[i, 0]:.2f},{locations[i, 1]:.2f},{locations[i, 2]:.2f}), "
        out = self.llm(prompt)
        return out





if __name__ == "__main__":
    env = LOS_Env(16,200)
    check_env(env, warn=True)

    policy_kwargs = dict(
        features_extractor_class=LLM_FE,
        features_extractor_kwargs=dict(features_dim=128),
    )
    model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(1000)