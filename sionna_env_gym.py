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

from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig




class sionna_env(gym.Env):

    def __init__(self,N,rng=42):
        super(sionna_env, self).__init__()
        self.rng = rng
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
        self.cm = self.scene.coverage_map(max_depth=2,
                                          diffraction=True,  # Disable to see the effects of diffraction
                                          cm_cell_size=(1., 1.),  # Grid size of coverage map cells in m
                                          combining_vec=None,
                                          precoding_vec=None,
                                          num_samples=int(2e6))  # Reduce if your hardware does not have enough memory

        cm_db = 10. * log10(self.cm._value[0, :, :])


        # Case-2
        # self.dens_poses = [
        #     [913, 1008],
        #     [889, 962],
        #     [896, 954],
        #     [903, 990],
        #     [872, 999],
        #     [899, 974],
        #     [878, 979],
        #     [890, 952],
        #     [886, 969],
        #     [880, 992]
        # ]

        # Case-1
        self.dens_poses = [
            (377, 612),
            (365, 573),
            (348, 581),
            (352, 584),
            (339, 585),
            (331, 566),
            (370, 613),
            (338, 588),
            (340, 612),
            (336, 599),
        ]

        self.rnd_poses = [
            (942,842),
            (621,  681),
            (731,  100),
            (537,  273),
            (403, 1215),
            (1098,  806)
        ]



        # Sample batch_size random positions
        self.ue_pos_dens = tf.gather_nd(self.cm.cell_centers, self.dens_poses)
        self.ue_pos_rnd = tf.gather_nd(self.cm.cell_centers, self.rnd_poses)

        for i in range(len(self.ue_pos_dens)):
            rx = Receiver(name=f"rx-{i}-dens",
                          position=self.ue_pos_dens[i],  # Random position sampled from coverage map
                          color=[0, 0, 1])
            self.scene.add(rx)

        for i in range(len(self.ue_pos_rnd)):
            rx = Receiver(name=f"rx-{i}-rnd",
                          position=self.ue_pos_rnd[i],  # Random position sampled from coverage map
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

        self.cm = self.scene.coverage_map(max_depth=2,
                                          diffraction=True,  # Disable to see the effects of diffraction
                                          cm_cell_size=(1., 1.),  # Grid size of coverage map cells in m
                                          combining_vec=None,
                                          precoding_vec=None,
                                          num_samples=int(2e6))  # Reduce if your hardware does not have enough memory

    def get_rssi(self):
        cm_db = 10. * log10(self.cm._value[0, :, :])
        cm_db = tf.where(cm_db < -400, -400, cm_db)
        rssi = tf.gather_nd(cm_db, self.dens_poses+self.rnd_poses)
        return rssi.numpy().astype(np.float32)

    def get_cm_db(self):
        cm_db = 10. * log10(self.cm._value)
        cm_db = tf.where(cm_db < -400, -400, cm_db)
        return cm_db.numpy().astype(np.float32)

    def get_prompt(self):
        locations = tf.concat([self.ue_pos_dens,self.ue_pos_rnd],0)
        idxs = self.dens_poses + self.rnd_poses
        cm_db = self.get_cm_db()[0]
        prompt = ""
        for i, loc in enumerate(locations):
            prompt += f"User at location ({locations[i, 0]:.2f},{locations[i, 1]:.2f},{locations[i, 2]:.2f}) gets {cm_db[tuple(idxs[i])]:.2f} dB signal power, "
        prompt+="Maximize the received signal power for all users."
        return prompt

    def get_prompt_neg(self):
        cm_db = self.get_cm_db()[0]
        prompt = ""
        for i, loc in enumerate(self.dens_poses):
            prompt += f"User at location ({self.ue_pos_dens[i, 0]:.2f},{self.ue_pos_dens[i, 1]:.2f},{self.ue_pos_dens[i, 2]:.2f}) gets {cm_db[tuple(loc)]:.2f} dB signal power, "

        for i, loc in enumerate(self.rnd_poses):
            prompt += f"User at location ({self.ue_pos_rnd[i, 0]:.2f},{self.ue_pos_rnd[i, 1]:.2f},{self.ue_pos_rnd[i, 2]:.2f}) gets {cm_db[tuple(loc)]:.2f} dB signal power, "
        prompt += "Maximize the received signal power for the following first 10 users while minimizing the received signal for the last 6 users."
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
    print(env.get_prompt_neg())
    # env.visualize('./',0)
    # tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    # encodings = tokenizer(prompt, max_length=512, padding=True, truncation=True, return_tensors="pt")
    # encodings = tokenizer(prompt, return_tensors="pt")

    # rssi = env.get_rssi()
    # check_env(env, warn=True)
    # policy_kwargs = dict(
    #     features_extractor_class=CustomCNN,
    #     features_extractor_kwargs=dict(features_dim=128),
    # )
    # model = SAC("CnnPolicy", env=env, policy_kwargs=policy_kwargs, verbose=1)
    # model.learn(10)