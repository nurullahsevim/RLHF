import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig



class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=128,
            name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)
        # self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # self.q = nn.Linear(self.fc2_dims, 1)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(2, 4, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with T.no_grad():
            n_flatten = self.cnn(
                T.zeros(1,1,1206,1476).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, fc1_dims), nn.ReLU(),nn.Linear(fc1_dims, fc2_dims),nn.ReLU(),)

        # self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.q = nn.Linear(self.fc2_dims+n_actions, 1)

        self.optimizer = optim.AdamW(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        action_value = self.cnn(state)
        action_value = self.linear(action_value)
        q = self.q(T.cat([action_value, action], dim=1))

        # action_value = self.fc1(T.cat([state, action], dim=1))
        # action_value = F.relu(action_value)
        # action_value = self.fc2(action_value)
        # action_value = F.relu(action_value)

        # q = self.q(action_value)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=128,
            name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        # self.v = nn.Linear(self.fc2_dims, 1)

        # self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(2, 4, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with T.no_grad():
            n_flatten = self.cnn(
                T.zeros(1,1,1206,1476).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, fc1_dims), nn.ReLU(),nn.Linear(fc1_dims, fc2_dims),nn.ReLU(),)

        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.AdamW(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        # state_value = self.fc1(state)
        # state_value = F.relu(state_value)
        # state_value = self.fc2(state_value)
        # state_value = F.relu(state_value)
        state_value = self.cnn(state)
        state_value = self.linear(state_value)
        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, model_name, alpha, input_dims, max_action, fc1_dims=256,
            fc2_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.model_name = model_name
        self.input_dims = input_dims
        # self.fc1_dims = fc1_dims
        # self.fc2_dims = fc2_dims
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(self.model_name, config=config)
        self.config = self.transformer.config
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.reparam_noise = 1e-6

        # self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(config.hidden_size, self.n_actions)
        self.sigma = nn.Linear(config.hidden_size, self.n_actions)

        self.optimizer = optim.AdamW(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, prompt):
        encodings = self.tokenizer(prompt, max_length=512, padding=True, truncation=True, return_tensors="pt")
        input_ids = encodings['input_ids'].to(self.device)
        attention_masks = encodings['attention_mask'].to(self.device)
        outputs = self.transformer(input_ids, attention_mask=attention_masks, output_hidden_states=True)
        last_hidden_states = outputs.last_hidden_state
        prob = last_hidden_states[:, 0, :]  # Taking the [CLS] token's representation
        # prob = self.fc1(state)
        prob = F.relu(prob)
        # prob = self.fc2(prob)
        # prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = F.relu(self.sigma(prob))

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, prompt, reparameterize=True):
        mu, sigma = self.forward(prompt)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        # log_probs -= T.log(1-actions.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))