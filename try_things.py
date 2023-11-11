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
from trlx.examples.randomwalks import generate_random_walks


if __name__ == '__main__':
    metric_fn, eval_prompts, sample_walks, logit_mask = generate_random_walks(42)
    pass
