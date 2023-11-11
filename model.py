from transformers import AutoModel, AutoConfig
import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self, model_name,output_num):
        super(RegressionModel, self).__init__()
        model_name = model_name
        config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.regression_head = nn.Linear(config.hidden_size, output_num)  # Assuming the output is a single scalar

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        pooled_output = last_hidden_states[:, 0, :]  # Taking the [CLS] token's representation
        return self.regression_head(pooled_output)

if __name__ == '__main__':
    model = RegressionModel('mistralai/Mistral-7B-v0.1',2)
    input = "Give two random numbers."
    output = model(input)