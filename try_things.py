import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from generate_data import MyDataset
from datasets import Dataset
from datasets import IterableDataset
import torch
import os,sys
from model import RegressionModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import AdamW
import torch.nn as nn
from sklearn.model_selection import train_test_split
import tqdm
import numpy as np
from wireless import LOS_Env
from transformers import pipeline, set_seed
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoModel, pipeline
import sionna
import tensorflow as tf
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

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    # Avoid warnings from TensorFlow
    tf.get_logger().setLevel('ERROR')
    tf.random.set_seed(1)  # Set global random seed for reproducibility

    # scene = load_scene('mitsuba/campus.xml')  # Try also sionna.rt.scene.etoile
    scene = load_scene(sionna.rt.scene.munich)
    scene.preview()
    scene.render(camera="preview", num_samples=512)