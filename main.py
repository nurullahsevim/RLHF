import os,sys

sys.path.insert(1, r'C:\Users\nurullahsevim\OneDrive - Texas A&M University\Desktop\research\RLHF\ddpg')
from generate_data import MyDataset
from datasets import Dataset
from datasets import IterableDataset
import torch
from ddpg_torch_llm import LLM_Agent
from ddpg_torch_cnn import CNN_Agent
from ddpg_torch_llmandcnn import LLMandCNN_Agent
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


def train(agent_type,log_dir,episode_length=1500,rng=42,visualize=True):




    if agent_type=='llm':
        agent = LLM_Agent(model_name, alpha=0.0005, beta=0.005, input_dims=[1, 1206, 1476], tau=0.001, env=None,
                              batch_size=8, layer1_size=256, layer2_size=128, n_actions=6,rng=rng)
    elif agent_type=='cnn':
        agent = CNN_Agent(model_name, alpha=0.0005, beta=0.005, input_dims=[1, 1206, 1476], tau=0.001, env=None,
                              batch_size=8, layer1_size=256, layer2_size=128, n_actions=6,rng=rng)
    else:
        agent = LLMandCNN_Agent(model_name, alpha=0.0005, beta=0.005, input_dims=[1, 1206, 1476], tau=0.001,
                                         env=None, batch_size=8, layer1_size=256, layer2_size=128, n_actions=6,rng=rng)

    log_dir = os.path.join(log_dir, agent_type)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    rewards_dir = os.path.join(log_dir,'rewards')
    if not os.path.exists(rewards_dir):
        os.mkdir(rewards_dir)


    if visualize:
        figs_dir = os.path.join(log_dir,'figs')
        if not os.path.exists(figs_dir):
            os.mkdir(figs_dir)


        figs_dir = os.path.join(figs_dir, str(rng))
        if not os.path.exists(figs_dir):
            os.mkdir(figs_dir)

    env = sionna_env(16)

    rewards = []

    for step_num in range(episode_length):
        prompt = env.get_prompt()
        obs = env.get_cm_db()/400
        obs_frw = torch.tensor(obs, dtype=torch.float).to(device)
        action = agent.choose_action(obs_frw,prompt)
        yaw = action[3]*180
        pitch = action[4]*90
        roll = action[5]*180
        env.initialize_transmitter(action[:2]*500,action[-2]*50+70,(yaw,pitch,roll))
        new_prompt = env.get_prompt()
        obs_ = env.get_cm_db()/400
        rssi = env.get_rssi()
        reward = np.mean(rssi)/40
        rewards.append(reward)
        agent.remember(obs, prompt, action, reward, obs_,new_prompt,False)
        if visualize:
            env.visualize(figs_dir,step_num)
        agent.learn()
        plt.plot(rewards)
        plt.savefig(rewards_dir + f"/rewards"+str(rng)+".png", dpi=300)
        plt.close()

    rewards = np.array(rewards)
    np.save(rewards_dir + f"/rewards"+str(rng)+".png",rewards)





if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"There are {torch.cuda.device_count()} GPU(s) available.")
        print("Device name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available. Using CPU instead.")
        device = torch.device("cpu")
    model_name = 'distilbert-base-uncased' #'bert-base-uncased' #"google/flan-t5-base"
    episode_length = 1500
    total_episodes = 1
    test_eps = 5

    rngs = [5,13,78,167,357]

    log_dir = f'../logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_dir = f'../logs/ddpg'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_dir = f'../logs/ddpg/{model_name}'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_dir = f'../logs/ddpg/{model_name}/run51'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    for rng in rngs:
        train('llm' ,log_dir, episode_length=episode_length, rng=rng, visualize=True)
        train('cnn',log_dir,  episode_length=episode_length, rng=rng, visualize=True)
        train('combined',log_dir,  episode_length=episode_length, rng=rng, visualize=True)

