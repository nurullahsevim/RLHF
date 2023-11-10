import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
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
from transformers import pipeline, set_seed
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


if __name__ == '__main__':
    train_data = torch.load('train.pt')
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto",
                                                       torch_dtype=torch.float16,max_length=512)

    input_text = "What is the BER for flat Rayleigh fading with QPSK?"
    print(input_text)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0]))
