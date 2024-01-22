import os,sys

sys.path.insert(1, r'C:\Users\Nurullah\Desktop\TAMU\Research\RLHF\ddpg')
from generate_data import MyDataset
from datasets import Dataset
from datasets import IterableDataset
import torch
from ddpg_torch_nollm import Agent
import matplotlib.pyplot as plt
from model import LLM
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import AdamW,Adam
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sionna_env_gym import sionna_env
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

    agent = Agent(model_name,alpha=0.001, beta=0.01, input_dims=[1,1206,1476], tau=0.001, env=None,
                  batch_size=4, layer1_size=256, layer2_size=128, n_actions=6)
    episode_length = 500
    total_episodes = 1
    test_eps = 5

    log_dir = f'../logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_dir = f'../logs/ddpg'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_dir = f'../logs/ddpg/{model_name}'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    model_checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoint_dir):
        os.mkdir(model_checkpoint_dir)

    figs_dir = os.path.join(log_dir, 'figs/run47_nollm')
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

    env = sionna_env(16)
    for episode in range(total_episodes):

        if not os.path.exists(os.path.join(trainfigs_dir,f'{episode}')):
            os.mkdir(os.path.join(trainfigs_dir,f'{episode}'))

        rewards = []
        score = 0
        best_reward = -10
        best_action = []
        for step_num in range(episode_length):
            prompt = env.get_prompt()
            obs = env.get_cm_db()/400
            obs_frw = torch.tensor(obs, dtype=torch.float).to(device)
            action = agent.choose_action(obs_frw)
            yaw = action[3]*180
            pitch = action[4]*90
            roll = action[5]*180
            env.initialize_transmitter(action[:2]*500,action[-2]*50+70,(yaw,pitch,roll))
            new_prompt = env.get_prompt()
            obs_ = env.get_cm_db()/400
            # observation_, reward, done, info = env.step(action)
            rssi = env.get_rssi()
            reward = np.mean(rssi)/40
            rewards.append(reward)
            score += reward/episode_length
            agent.remember(obs, prompt, action, reward, obs_,new_prompt,False)
            env.visualize(os.path.join(trainfigs_dir,f'{episode}'),step_num)
            agent.learn()
            plt.plot(rewards)
            plt.savefig(os.path.join(trainfigs_dir, f'{episode}') + f"/rewards.png", dpi=300)
            plt.close()
            if reward>best_reward:
                best_reward = reward
                best_action = [action[:2]*500,action[-2]*50+70,(yaw,pitch,roll)]
        print('Best reward: ', best_reward)
        print('Best Action: ', best_action)
        # plt.plot(rewards)
        # plt.savefig(os.path.join(trainfigs_dir,f'{episode}') + f"/rewards.png", dpi=300)
        # plt.close()