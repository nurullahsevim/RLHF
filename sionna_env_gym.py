import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os,sys
from gymnasium import spaces
import string
from stable_baselines3.common.env_checker import check_env
import torch as T
import torch.nn as nn
from model import LLM
from stable_baselines3 import PPO,SAC,DDPG
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import sionna
import tensorflow as tf
import time

# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera
from sionna.utils import expand_to_rank, log10




class sionna_env(gym.Env):

    def __init__(self,N):
        super(sionna_env, self).__init__()
        self.n_receivers = N
        self.step_num = 0
        self.terminal_step = 4

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )

        self.observation_space = spaces.Box(low=0, high=1, shape=(1,1206, 1476), dtype=np.float32)

        self.scene = load_scene(sionna.rt.scene.munich)
        size = self.scene.size
        center = self.scene.center
        bird_pos = [center[0], center[1] - 0.01, 3000]

        # Create new camera
        bird_cam = Camera("birds_view", position=bird_pos, look_at=center)
        self.scene.add(bird_cam)

        # define scene borders
        self.right = (size / 2 + center)[0] - 50
        self.left = (- size / 2 + center)[0] + 50
        self.top = (size / 2 + center)[1] - 50
        self.bottom = (- size / 2 + center)[1] + 50

        # define transmitter and receiver antenna arrays
        self.scene.tx_array = PlanarArray(num_rows=8,
                                          num_cols=2,
                                          vertical_spacing=0.7,
                                          horizontal_spacing=0.5,
                                          pattern="tr38901",
                                          polarization="VH")

        # Configure antenna array for all receivers
        self.scene.rx_array = PlanarArray(num_rows=1,
                                          num_cols=1,
                                          vertical_spacing=0.5,
                                          horizontal_spacing=0.5,
                                          pattern="dipole",
                                          polarization="cross")

        # define the transmitter with randomly initialized location
        tx = Transmitter(name="tx",
                         position=[center[0], center[0], 50],
                         orientation=[0, 0, 0])
        self.scene.add(tx)

        # choose random locations for receivers
        self.cm = self.scene.coverage_map(max_depth=5,
                                          diffraction=True,  # Disable to see the effects of diffraction
                                          cm_cell_size=(1., 1.),  # Grid size of coverage map cells in m
                                          combining_vec=None,
                                          precoding_vec=None,
                                          num_samples=int(1e6))  # Reduce if your hardware does not have enough memory

        cm_db = 10. * log10(self.cm._value[0, :, :])
        idx = tf.where(tf.math.logical_and(cm_db > -200,
                                           cm_db < 0))
        # Randomly permute indices
        idx = tf.random.shuffle(idx)
        self.rx_idx = idx[:N]

        # Sample batch_size random positions
        self.ue_pos = tf.gather_nd(self.cm.cell_centers, self.rx_idx)

        for i in range(N):
            rx = Receiver(name=f"rx-{i}",
                          position=self.ue_pos[i],  # Random position sampled from coverage map
                          color=[1, 0, 0])
            self.scene.add(rx)

    def initialize_transmitter(self,loc):
        if loc[0] < self.left:
            tr_loc_x = self.left
        elif loc[0] > self.right:
            tr_loc_x = self.right
        else:
            tr_loc_x = loc[0]

        if loc[1] < self.bottom:
            tr_loc_y = self.bottom
        elif loc[1] > self.top:
            tr_loc_y = self.top
        else:
            tr_loc_y = loc[1]

        self.scene.remove("tx")

        tx = Transmitter(name="tx",
                         position=[tr_loc_x, tr_loc_y, 50],
                         orientation=[0, 0, 0])

        self.scene.add(tx)

        self.cm = self.scene.coverage_map(max_depth=5,
                                          diffraction=True,  # Disable to see the effects of diffraction
                                          cm_cell_size=(1., 1.),  # Grid size of coverage map cells in m
                                          combining_vec=None,
                                          precoding_vec=None,
                                          num_samples=int(1e6))  # Reduce if your hardware does not have enough memory

    def get_rssi(self):
        cm_db = 10. * log10(self.cm._value[0, :, :])
        cm_db = tf.where(cm_db < -200, -200, cm_db)
        rssi = tf.gather_nd(cm_db, self.rx_idx)
        return rssi.numpy().astype(np.float32)

    def get_cm_db(self):
        cm_db = 10. * log10(self.cm._value)
        cm_db = tf.where(cm_db < -200, -200, cm_db)
        return cm_db.numpy().astype(np.float32)

    def get_prompt(self):
        locations = self.ue_pos
        prompt = ""
        for i, loc in enumerate(locations):
            prompt += f"Location {i + 1}: ({locations[i, 0]:.2f},{locations[i, 1]:.2f},{locations[i, 2]:.2f}), "
        return prompt

    def reset(self, seed=None, options=None):

        super().reset(seed=seed, options=options)
        self.step_num = 0
        tr_loc = np.random.uniform(low=-500, high=500, size=(2,))  # randomly choosing a location for transmitter
        self.initialize_transmitter(tr_loc)  # initialize one transmitter later should convert this to a list
        obs = self.get_cm_db()/200+1
        return obs, {}  # empty info dict

    def step(self, action):
        self.step_num += 1
        self.initialize_transmitter(500*action)  # initialize one transmitter later should convert this to a list
        rssi = self.get_rssi()


        reward = np.mean(rssi).astype(float)
        terminated = False if self.step_num<self.terminal_step else True
        truncated = True
        obs = self.get_cm_db() / 200 + 1
        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        self.scene.render("birds_view",coverage_map=self.cm, show_devices=True, num_samples=256)

    def close(self):
        pass



class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 1, kernel_size=8, stride=8, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with T.no_grad():
            n_flatten = self.cnn(
                T.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: T.Tensor) -> T.Tensor:
        return self.linear(self.cnn(observations))




if __name__ == "__main__":
    env = sionna_env(16)
    # rssi = env.get_rssi()
    # check_env(env, warn=True)
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    model = SAC("CnnPolicy", env=env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(10)