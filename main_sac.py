import os,sys

sys.path.insert(1, '/home/nurullah/Research/RLHF/sac')
from generate_data import MyDataset
from datasets import Dataset
from datasets import IterableDataset
import torch
from sac_torch import Agent
import matplotlib.pyplot as plt
from model import RegressionModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import AdamW,Adam
import torch.nn as nn
from utils import plot_learning_curve
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from wireless import LOS_Env
from transformers import AutoModelForSequenceClassification


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"There are {torch.cuda.device_count()} GPU(s) available.")
        print("Device name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available. Using CPU instead.")
        device = torch.device("cpu")
    model_name = 'distilbert-base-uncased' #'bert-base-uncased' #"google/flan-t5-base"

    agent = Agent(model_name,batch_size=8)
    episode_length = 1
    total_episodes = 4000
    test_eps = 100

    log_dir = f'logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_dir = f'logs/sac'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_dir = f'logs/sac/{model_name}'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    model_checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoint_dir):
        os.mkdir(model_checkpoint_dir)

    figs_dir = os.path.join(log_dir, 'figs/run8')
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)

    trainfigs_dir = os.path.join(figs_dir, 'train')
    if not os.path.exists(trainfigs_dir):
        os.makedirs(trainfigs_dir)

    testfigs_dir = os.path.join(figs_dir, 'test')
    if not os.path.exists(testfigs_dir):
        os.makedirs(testfigs_dir)

    score_history = []
    best_score = 0
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    for episode in range(total_episodes):

        if not os.path.exists(os.path.join(trainfigs_dir,f'{episode}')):
            os.mkdir(os.path.join(trainfigs_dir,f'{episode}'))

        mean = np.random.uniform(low=-250,high=250)
        env = LOS_Env(16, mean, device)
        prompt = env.get_prompt()
        score = 0
        for step_num in range(episode_length):
            rssi = env.get_rssi()
            action = agent.choose_action(prompt)
            loc = np.append(action, np.array([50]))
            env.initialize_transmitter(loc)
            rssi_ = env.get_rssi()
            # observation_, reward, done, info = env.step(action)
            reward = np.sum(rssi_)/env.n_receivers
            score += reward
            agent.remember(rssi, prompt, action, reward, rssi_)
            env.visualize(os.path.join(trainfigs_dir,f'{episode}'),step_num)
            if not load_checkpoint:
                agent.learn()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', episode, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(total_episodes)]
        plot_learning_curve(x, score_history, os.path.join(figs_dir,"learning_curve.png"))