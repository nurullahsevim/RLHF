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



if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"There are {torch.cuda.device_count()} GPU(s) available.")
        print("Device name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available. Using CPU instead.")
        device = torch.device("cpu")
    model_name = 'distilbert-base-uncased' #'bert-base-uncased' #"google/flan-t5-base"
    model = RegressionModel(model_name,24*8)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # for param in model.transformer.parameters():
    #     param.requires_grad = False
    model = model.to(device)

    train_data = torch.load('train.pt')
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=16)


    optimizer = AdamW(model.parameters(), lr=1e-5)
    tol = 0.01
    last_dif = 10
    # Training loop
    for epoch in range(150):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            prompts, labels = batch
            encodings = tokenizer(prompts, max_length=512, padding=True, truncation=True, return_tensors="pt")
            input_ids = encodings['input_ids'].to("cuda")
            attention_masks = encodings['attention_mask'].to("cuda")
            labels = torch.tensor(labels, dtype=torch.float32)
            labels = labels.to(device)
            # Forward pass
            outputs = model(input_ids,attention_masks)
            loss = nn.MSELoss()(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("Epoch:", epoch, "Loss:", total_loss / len(train_dataloader))
    torch.save(model.state_dict(), "demo_model.pth")