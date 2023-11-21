from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import torch


class RegressionModel(nn.Module):
    def __init__(self, model_name,output_num):
        super(RegressionModel, self).__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(self.model_name, config=config)
        self.config = self.transformer.config
        self.regression_head = nn.Linear(config.hidden_size, output_num)  # Assuming the output is a single scalar
        self.head_activation = nn.Tanh()

    def forward(self, input_ids, attention_mask, **kwargs):
        kwargs = kwargs
        outputs = self.transformer(input_ids, attention_mask=attention_mask,output_hidden_states=True)
        last_hidden_states = outputs.last_hidden_state
        pooled_output = last_hidden_states[:, 0, :]  # Taking the [CLS] token's representation
        regression_output = self.regression_head(pooled_output)
        regression_output = self.head_activation(regression_output)
        # outputs.last_hidden_state = self.head_activation(regression_output)
        # outputs.hidden_states = outputs.hidden_states + (self.head_activation(regression_output),)
        return regression_output


    def generate(self, input_ids, attention_mask=None, **kwargs):
        kwargs = kwargs
        output = self.forward(input_ids, attention_mask=None)
        output_np = output.detach().numpy()
        output_str = f"{output_np[0]:.3f},{output_np[1]:.3f}"
        encoded_output = self.tokenizer(output_str, return_tensors='pt')
        return encoded_output
