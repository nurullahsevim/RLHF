from generate_data import MyDataset
from datasets import Dataset
from datasets import IterableDataset
import torch
import matplotlib.pyplot as plt
from model_tf import LLM
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import AdamW,Adam
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os,sys
from sionna_env import sionna_env
from transformers import AutoModelForSequenceClassification
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer



if __name__ == '__main__':
    if tf.test.is_gpu_available():
        device = "cuda"
        print(f"There are {len(tf.config.experimental.list_physical_devices('GPU'))} GPU(s) available.")
        print("Device name:", tf.test.gpu_device_name())
    else:
        print("CUDA is not available. Using CPU instead.")
        device = "cpu"

    model_name = 'distilbert-base-uncased'  # 'bert-base-uncased' #"google/flan-t5-base"
    model = LLM(model_name, 2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-6)
    episode_length = 100
    total_episodes = 1
    test_eps = 5

    log_dir = '../logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_dir = f'../logs/{model_name}'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    model_checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoint_dir):
        os.mkdir(model_checkpoint_dir)

    figs_dir = os.path.join(log_dir, 'figs/run30')
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)

    trainfigs_dir = os.path.join(figs_dir, 'train')
    if not os.path.exists(trainfigs_dir):
        os.makedirs(trainfigs_dir)

    testfigs_dir = os.path.join(figs_dir, 'test')
    if not os.path.exists(testfigs_dir):
        os.makedirs(testfigs_dir)


    def loss_fn(rssi):
        # Assuming 'rssi' is a tensor computed outside of this function
        return -tf.reduce_sum(rssi) / env.n_receivers


    def eval(model, test_eps):
        for episode in tqdm(range(test_eps)):
            env = sionna_env(16)
            total_loss = 0
            prompt = env.get_prompt()
            encodings = tokenizer(prompt, max_length=512, padding=False, truncation=True, return_tensors="tf")
            input_ids = tf.convert_to_tensor(encodings['input_ids'])
            attention_mask = tf.convert_to_tensor(encodings['attention_mask'])
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            pred = model(inputs)

            # Handling 'env' logic outside loss function
            loc = 600 * tf.squeeze(pred)
            env.initialize_transmitter(loc)
            rssi = env.get_rssi()

            # Compute the loss using only tensors
            loss = loss_fn(rssi)
            total_loss += loss.numpy()  # Accumulate loss over steps

            # Your visualization logic here, if needed
            env.visualize(testfigs_dir,episode)


    # Training loop
    reward_var = []
    for episode in range(total_episodes):
        if not os.path.exists(os.path.join(trainfigs_dir, f'{episode}')):
            os.mkdir(os.path.join(trainfigs_dir, f'{episode}'))

        model.compile(optimizer=optimizer, loss=loss_fn)
        total_loss = 0
        env = sionna_env(64)
        episode_rewards = np.zeros((0,))
        env.visualize(os.path.join(trainfigs_dir, f'{episode}'), 0)
        for step_num in tqdm(range(episode_length)):

            # Update the model's weights using gradient tape or other TensorFlow methods
            with tf.GradientTape() as tape:
                prompt = env.get_prompt()
                encodings = tokenizer(prompt, max_length=512, padding=False, truncation=True, return_tensors="tf")
                input_ids = tf.convert_to_tensor(encodings['input_ids'])
                attention_mask = tf.convert_to_tensor(encodings['attention_mask'])
                inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
                predictions = model(inputs, training=True)
                print(predictions)
                loc = 500 * tf.squeeze(predictions)
                env.initialize_transmitter(loc)
                rssi = env.get_rssi()
                loss = loss_fn(rssi)  # Assuming get_rssi returns a tensor
                print(loss)
                env.visualize(os.path.join(trainfigs_dir, f'{episode}'), step_num+1)
                episode_rewards = np.append(episode_rewards, -loss.numpy())
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        reward_var.append(np.var(episode_rewards))
        print("Episode:", episode, "Initial Reward: ", episode_rewards[0], "Last Reward:", episode_rewards[-1])
    plt.plot(reward_var)
    plt.savefig(trainfigs_dir + "/vars.png", dpi=300)
    plt.close()
    model.save_weights(os.path.join(model_checkpoint_dir, 'tf_model_weights.h5'))
    tokenizer.save_vocabulary(model_checkpoint_dir)

    eval(model, test_eps)