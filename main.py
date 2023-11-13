from generate_data import MyDataset
from datasets import Dataset
from datasets import IterableDataset
import torch
import matplotlib.pyplot as plt
from model import RegressionModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import AdamW
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

    # for param in model.transformer.parameters():
    #     param.requires_grad = False
    model = model.to(device)

    # train_data = torch.load('train.pt')
    # train_sampler = RandomSampler(train_data)
    # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=16)


    optimizer = AdamW(model.parameters(), lr=3e-5)
    episode_length = 16
    total_episodes = 1000

    log_dir = f'logs/{model_name}'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    model_checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoint_dir):
        os.mkdir(model_checkpoint_dir)

    figs_dir = os.path.join(log_dir, 'figs/run4')
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)

    def loss_fn(env,output):
        loc = 500*output.squeeze()
        loc = torch.cat((loc,torch.tensor([50]).to(loc.device)))
        env.initialize_transmitter(loc)
        rssi = env.get_rssi()
        return -torch.sum(rssi)/env.n_receivers

    # Training loop
    loss_var = []
    for episode in range(total_episodes):

        if not os.path.exists(os.path.join(figs_dir,f'{episode}')):
            os.mkdir(os.path.join(figs_dir,f'{episode}'))

        model.train()
        total_loss = 0
        env = LOS_Env(16,device)
        episode_losses = np.zeros((0,))
        for step_num  in tqdm(range(episode_length)):
            prompt = env.get_prompt()
            encodings = tokenizer(prompt, max_length=512, padding=False, truncation=True, return_tensors="pt")
            input_ids = encodings['input_ids'].to("cuda")
            attention_masks = encodings['attention_mask'].to("cuda")
            # labels = torch.tensor(labels, dtype=torch.float32)
            # labels = labels.to(device)
            # Forward pass
            pred = model(input_ids,attention_masks).last_hidden_state
            loss = loss_fn(env,pred)
            env.visualize(os.path.join(figs_dir,f'{episode}'),step_num)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            episode_losses = np.append(episode_losses, -loss.to("cpu").detach().numpy())

        loss_var.append(np.var(episode_losses))
        print("Episode:", episode, "Initial Loss: ",episode_losses[0], "Last Loss:", episode_losses[-1])
    plt.plot(loss_var)
    plt.savefig(figs_dir, dpi=300)
    plt.close()
    # torch.save(model.state_dict(), os.path.join(model_checkpoint_dir, 'pytorch_model.bin'))
    # torch.save(optimizer.state_dict(), os.path.join(model_checkpoint_dir, 'optimizer.pt'))
    # tokenizer.save_vocabulary(model_checkpoint_dir)