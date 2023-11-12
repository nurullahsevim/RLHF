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
from transformers import pipeline, set_seed
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoModel, pipeline
from trlx.examples.randomwalks import generate_random_walks


def foo(a,b,c):
    return a+b+c


if __name__ == '__main__':
    model_name = "distilgpt2"
    model = RegressionModel(model_name, 2)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)

    output_dir = "./models/"

    # Step 1: Save a model, configuration and vocabulary that you have fine-tuned

    # If we have a distributed model, save only the encapsulated model
    # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
    model_to_save = model.module if hasattr(model, 'module') else model

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_dir, 'pytorch_model.bin')
    output_config_file = os.path.join(output_dir, 'config.json')

    torch.save(model.state_dict(), output_model_file)
    model.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)
    # load again
    # Example for a Bert model
    model = AutoModel.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)








    # model = AutoModel.from_pretrained("model.pth")

    # model = RegressionModel('distilbert-base-uncased',2)
    # torch.save(model,'trlx/models/model.pth')

    # model = torch.load('model.pth')
    # print(model.model_name)
    # tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    # model = AutoModel.from_pretrained("distilbert-base-uncased")
    # text = "Output 2."
    # encoded_input = tokenizer(text, return_tensors='pt')
    # output = model(**encoded_input)
    # pass
    # print(output.detach().numpy())
    # output_np = output.detach().numpy()
    # output_str = f"{output_np[0]:.3f},{output_np[1]:.3f}"
    # encoded_output = tokenizer(output_str, return_tensors='pt')
    # print(tokenizer.decode(encoded_output['input_ids'].squeeze(), skip_special_tokens=True))
    # foo2 = lambda a,b: foo(5,a,b)
    # print(foo2(1,2))

