# import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from generate_data import MyDataset
# from datasets import Dataset
# from datasets import IterableDataset
# import torch
# import os,sys
# from model import RegressionModel
# from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler
# from torch.optim import AdamW
# import torch.nn as nn
# from sklearn.model_selection import train_test_split
# import tqdm
# import numpy as np
# from wireless import LOS_Env
# from transformers import pipeline, set_seed
# import transformers
# from transformers import LlamaTokenizer, LlamaForCausalLM
# from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoModel, pipeline
import sionna
# import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera

# For link-level simulations
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement




if __name__ == '__main__':

    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         tf.config.experimental.set_memory_growth(gpus[0], True)
    #     except RuntimeError as e:
    #         print(e)
    # # Avoid warnings from TensorFlow
    # tf.get_logger().setLevel('ERROR')
    # tf.random.set_seed(1)  # Set global random seed for reproducibility

    scene = load_scene('mitsuba/campus_light/campus.xml')  # Try also sionna.rt.scene.etoile

    # Configure antenna array for all transmitters
    scene.tx_array = PlanarArray(num_rows=8,
                                 num_cols=2,
                                 vertical_spacing=0.7,
                                 horizontal_spacing=0.5,
                                 pattern="tr38901",
                                 polarization="VH")

    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=1,
                              num_cols=1,
                              vertical_spacing=0.5,
                              horizontal_spacing=0.5,
                              pattern="dipole",
                              polarization="cross")

    # Create transmitter
    tx = Transmitter(name="tx",
                  position=[8.5,21,27],
                  orientation=[0,0,0])
    scene.add(tx)

    # Create a receiver
    rx = Receiver(name="rx",
               position=[45,90,1.5],
               orientation=[0,0,0])
    scene.add(rx)

    # TX points towards RX
    tx.look_at(rx)

    print(scene.transmitters)
    print(scene.receivers)



    my_cam = Camera("my_cam", position=[8.5,21,500], look_at=[45,90,1.5])
    scene.add(my_cam)

    paths = scene.compute_paths(max_depth=5,
                                num_samples=1e6)
    scene.preview(paths=paths)  # Open preview showing paths
    scene.render(camera="preview", paths=paths)  # Render scene with paths from preview camera
    scene.render_to_file(camera=my_cam,
                         filename="scene_path3.png",
                         paths=paths)  # Render scene with paths to file



    cm = scene.coverage_map(cm_cell_size=[1.,1.], # Configure size of each cell
                           num_samples=1e7) # Number of rays to trace

    scene.preview(coverage_map=cm)  # Open preview showing coverage map
    scene.render(camera="preview", coverage_map=cm)  # Render scene with coverage map
    scene.render_to_file(camera=my_cam,
                         filename="scene_cm3.png",
                         coverage_map=cm)  # Render scene with coverage map to file