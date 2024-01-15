import sionna
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os,sys

# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera
from sionna.utils import expand_to_rank, log10

class sionna_env:
    def __init__(self,N):

        self.n_receivers = N
        #initialize scene and camera
        self.scene = load_scene(sionna.rt.scene.munich)
        size = self.scene.size
        center = self.scene.center
        bird_pos = [center[0], center[1] - 0.01, 3000]

        # Create new camera
        bird_cam = Camera("birds_view", position=bird_pos, look_at=center)
        self.scene.add(bird_cam)

        #define scene borders
        self.right = (size/2 + center)[0]-50
        self.left = (- size/2 + center)[0]+50
        self.top = (size/2 + center)[1]-50
        self.bottom = (- size/2 + center)[1]+50

        #define transmitter and receiver antenna arrays
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


        #define the transmitter with randomly initialized location
        tx = Transmitter(name="tx",
                         position=[center[0], center[0], 50],
                         orientation=[0, 0, 0])
        self.scene.add(tx)

        #choose random locations for receivers
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
                          color=[1,0,0])
            self.scene.add(rx)

    def initialize_transmitter(self, loc):
        if loc[0]<self.left:
            tr_loc_x = self.left
        elif loc[0]>self.right:
            tr_loc_x = self.right
        else:
            tr_loc_x = loc[0]

        if loc[1]<self.bottom:
            tr_loc_y = self.bottom
        elif loc[1]> self.top:
            tr_loc_y = self.top
        else:
            tr_loc_y = loc[1]

        tx = Transmitter(name="tx",
                         position=[tr_loc_x, tr_loc_y, 50],
                         orientation=[0, 0, 0])
        self.scene.remove("tx")
        self.scene.add(tx)

        self.cm = self.scene.coverage_map(max_depth=5,
                                          diffraction=True,  # Disable to see the effects of diffraction
                                          cm_cell_size=(1., 1.),  # Grid size of coverage map cells in m
                                          combining_vec=None,
                                          precoding_vec=None,
                                          num_samples=int(1e6))  # Reduce if your hardware does not have enough memory

    def get_rssi(self):
        cm_db = 10. * log10(self.cm._value[0, :, :])
        cm_db = tf.where(cm_db < -400, -400, cm_db)
        rssi = tf.gather_nd(cm_db, self.rx_idx)
        return rssi

    def get_state(self):
        cm_db = 10. * log10(self.cm._value[0, :, :])
        cm_db = tf.where(cm_db < -400, -400, cm_db)
        return cm_db

    def visualize(self, dir_path, fig_num):
        self.scene.render_to_file("birds_view",os.path.join(dir_path, f"{fig_num}.png"),coverage_map=self.cm, show_devices=True, num_samples=256,cm_vmax=0,cm_vmin=-400)

    def get_prompt(self):
        locations = self.ue_pos
        idxs = self.rx_idx
        cm_db = self.get_state()
        prompt = ""
        for i, loc in enumerate(locations):
            prompt += f"User at location ({locations[i, 0]:.2f},{locations[i, 1]:.2f},{locations[i, 2]:.2f}) gets {cm_db[tuple(idxs[i].numpy())]:.2f} dB signal power, "
            # prompt += f"Location {i + 1}: ({locations[i, 0]:.2f},{locations[i, 1]:.2f},{locations[i, 2]:.2f}), "
        return prompt


if __name__ == '__main__':
    env = sionna_env(16)
    print(env.get_prompt())
    env.visualize("./",0)
    env.get_rssi()
