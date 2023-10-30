from generate_data import MyDataset
from datasets import Dataset
from datasets import IterableDataset
import torch
from model import RegressionModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import AdamW
import torch.nn as nn
from sklearn.model_selection import train_test_split
import tqdm
import numpy as np

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"There are {torch.cuda.device_count()} GPU(s) available.")
        print("Device name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available. Using CPU instead.")
        device = torch.device("cpu")

    model_name = 'distilbert-base-uncased' #'bert-base-uncased' #"google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = RegressionModel(model_name,24*8)

    # Then load the state dictionary
    state_dict = torch.load('demo_model.pth')
    model.load_state_dict(state_dict)
    model = model.to(device)

    val_data = torch.load('val.pt')
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=1)

    with torch.no_grad():
        for batch in tqdm.tqdm(val_dataloader):
            prompts, labels = batch
            encodings = tokenizer(prompts, max_length=512, padding=True, truncation=True, return_tensors="pt")
            input_ids = encodings['input_ids'].to("cuda")
            attention_masks = encodings['attention_mask'].to("cuda")
            labels = torch.tensor(labels, dtype=torch.float32)
            labels = labels.to(device)
            # Forward pass
            outputs = model(input_ids, attention_masks)
            outputs_np = outputs.cpu().detach().numpy()
            outputs_np = np.array(outputs_np).astype(int)
            print(str(prompts).replace('\n','\n'))
            print(outputs_np.reshape(24,8))
