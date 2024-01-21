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
                         orientation=[0, 0, 0],
                         color=[0,0,0])
        self.scene.add(tx)

        # choose random locations for receivers
        self.cm = self.scene.coverage_map(max_depth=5,
                                          diffraction=True,  # Disable to see the effects of diffraction
                                          cm_cell_size=(5., 5.),  # Grid size of coverage map cells in m
                                          combining_vec=None,
                                          precoding_vec=None,
                                          num_samples=int(1e7))  # Reduce if your hardware does not have enough memory

        cm_db = 10. * log10(self.cm._value[0, :, :])


        dens_poses = [
            [740, 800],
            [750, 810],
            [760, 820],
            [770, 830],
            [780, 840],
            [740, 850],
            [750, 860],
            [760, 870],
            [770, 880],
            [750, 840]
        ]

        dens_poses2 = [
            [880, 940],
            [880, 950],
            [880, 960],
            [880, 970],
            [880, 980],
            [900, 940],
            [900, 950],
            [900, 960],
            [900, 970],
            [900, 980]
        ]

        dens_poses_cellsize5 = [
            [70, 110],
            [70, 113],
            [70, 116],
            [70, 119],
            [70, 122],
            [74, 110],
            [74, 113],
            [74, 116],
            [74, 119],
            [74, 122],
        ]

        idx = tf.where(tf.math.logical_and(cm_db > -400,
                                           cm_db < 0))
        # Randomly permute indices
        idx = tf.random.shuffle(idx)
        self.rx_idx = idx[:N - 10].numpy().tolist() + dens_poses_cellsize5


        # Sample batch_size random positions
        self.ue_pos = tf.gather_nd(self.cm.cell_centers, self.rx_idx)

        for i in range(N):
            rx = Receiver(name=f"rx-{i}",
                          position=self.ue_pos[i],  # Random position sampled from coverage map
                          color=[1, 0, 0])
            self.scene.add(rx)


    def initialize_transmitter(self,loc,height,orientation=[0, 0, 0]):
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
                         position=[tr_loc_x, tr_loc_y, height],
                         orientation=orientation,
                         color=[0,0,0])

        self.scene.add(tx)

        self.cm = self.scene.coverage_map(max_depth=5,
                                          diffraction=True,  # Disable to see the effects of diffraction
                                          cm_cell_size=(5., 5.),  # Grid size of coverage map cells in m
                                          combining_vec=None,
                                          precoding_vec=None,
                                          num_samples=int(1e7))  # Reduce if your hardware does not have enough memory

    def get_rssi(self):
        cm_db = 10. * log10(self.cm._value[0, :, :])
        cm_db = tf.where(cm_db < -400, -400, cm_db)
        rssi = tf.gather_nd(cm_db, self.rx_idx)
        return rssi.numpy().astype(np.float32)

    def get_cm_db(self):
        cm_db = 10. * log10(self.cm._value)
        cm_db = tf.where(cm_db < -400, -400, cm_db)
        return cm_db.numpy().astype(np.float32)

    def get_prompt(self):
        locations = self.ue_pos
        idxs = self.rx_idx
        cm_db = self.get_cm_db()[0]
        prompt = ""
        for i, loc in enumerate(locations):
            prompt += f"User at location ({locations[i, 0]:.2f},{locations[i, 1]:.2f},{locations[i, 2]:.2f}) gets {cm_db[tuple(idxs[i])]:.2f} dB signal power, "
            # prompt += f"Location {i + 1}: ({locations[i, 0]:.2f},{locations[i, 1]:.2f},{locations[i, 2]:.2f}), "
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

    def visualize(self, dir_path, fig_num):
        self.scene.render_to_file("birds_view",os.path.join(dir_path, f"{fig_num}.png"),coverage_map=self.cm, show_devices=True, num_samples=256,cm_vmax=0,cm_vmin=-400)

    def close(self):
        pass



if __name__ == "__main__":
    env = sionna_env(16)
    print(env.get_prompt())
    # rssi = env.get_rssi()
    # check_env(env, warn=True)
    # policy_kwargs = dict(
    #     features_extractor_class=CustomCNN,
    #     features_extractor_kwargs=dict(features_dim=128),
    # )
    # model = SAC("CnnPolicy", env=env, policy_kwargs=policy_kwargs, verbose=1)
    # model.learn(10)