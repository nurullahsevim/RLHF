from generate_data import MyDataset
from datasets import Dataset
from datasets import IterableDataset
import torch
import matplotlib.pyplot as plt
from model import RegressionModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import AdamW,Adam
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os,sys
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
    # model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = RegressionModel(model_name,2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = model.to(device)


    optimizer = AdamW(model.parameters(), lr=3e-5)
    episode_length = 4
    total_episodes = 10000
    test_eps = 100

    log_dir = f'logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_dir = f'logs/{model_name}'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    model_checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoint_dir):
        os.mkdir(model_checkpoint_dir)

    figs_dir = os.path.join(log_dir, 'figs/run14')
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)

    trainfigs_dir = os.path.join(figs_dir, 'train')
    if not os.path.exists(trainfigs_dir):
        os.makedirs(trainfigs_dir)

    testfigs_dir = os.path.join(figs_dir, 'test')
    if not os.path.exists(testfigs_dir):
        os.makedirs(testfigs_dir)

    def loss_fn(env,output):
        loc = 500*output.squeeze()
        # loc = output.squeeze()
        loc = torch.cat((loc,torch.tensor([50]).to(loc.device)))
        env.initialize_transmitter(loc)
        rssi = env.get_rssi()
        return -torch.sum(rssi)/env.n_receivers

    def eval(model, test_eps):
        for episode in tqdm(range(test_eps)):
            model.eval()
            mean = 500 * torch.rand(1) - 250
            env = LOS_Env(16, mean, device)
            prompt = env.get_prompt()
            encodings = tokenizer(prompt, max_length=512, padding=False, truncation=True, return_tensors="pt")
            input_ids = encodings['input_ids'].to("cuda")
            attention_masks = encodings['attention_mask'].to("cuda")
            # labels = torch.tensor(labels, dtype=torch.float32)
            # labels = labels.to(device)
            # Forward pass
            pred = model(input_ids, attention_masks)
            env.visualize(os.path.join(testfigs_dir), step_num)

    # Training loop
    reward_var = []
    for episode in range(total_episodes):

        if not os.path.exists(os.path.join(trainfigs_dir,f'{episode}')):
            os.mkdir(os.path.join(trainfigs_dir,f'{episode}'))

        model.train()
        total_loss = 0
        mean = 500 * torch.rand(1) - 250
        env = LOS_Env(16,mean,device)
        episode_rewards = np.zeros((0,))
        for step_num  in tqdm(range(episode_length)):
            prompt = env.get_prompt()
            encodings = tokenizer(prompt, max_length=512, padding=False, truncation=True, return_tensors="pt")
            input_ids = encodings['input_ids'].to("cuda")
            attention_masks = encodings['attention_mask'].to("cuda")
            # labels = torch.tensor(labels, dtype=torch.float32)
            # labels = labels.to(device)
            # Forward pass
            pred = model(input_ids,attention_masks)
            loss = loss_fn(env,pred)
            env.visualize(os.path.join(trainfigs_dir,f'{episode}'),step_num)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            episode_rewards = np.append(episode_rewards, -loss.to("cpu").detach().numpy())

        reward_var.append(np.var(episode_rewards))
        print("Episode:", episode, "Initial Reward: ",episode_rewards[0], "Last Reward:", episode_rewards[-1])
    plt.plot(reward_var)
    plt.savefig(trainfigs_dir+f"/vars.png", dpi=300)
    plt.close()
    torch.save(model.state_dict(), os.path.join(model_checkpoint_dir, 'pytorch_model.bin'))
    torch.save(optimizer.state_dict(), os.path.join(model_checkpoint_dir, 'optimizer.pt'))
    tokenizer.save_vocabulary(model_checkpoint_dir)

    eval(model,test_eps)