from transformers import TFAutoModel, AutoConfig, AutoTokenizer
import tensorflow as tf

class LLM(tf.keras.Model):
    def __init__(self, model_name, output_num):
        super(LLM, self).__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.transformer = TFAutoModel.from_pretrained(self.model_name)
        self.regression_head = tf.keras.layers.Dense(output_num)  # Assuming the output is a single scalar
        self.activation = tf.keras.activations.tanh

    def call(self, encodings, **kwargs):
        input_ids = encodings['input_ids']
        attention_masks = encodings['attention_mask']
        outputs = self.transformer(input_ids, attention_mask=attention_masks, output_hidden_states=True)
        last_hidden_states = outputs.last_hidden_state
        pooled_output = last_hidden_states[:, 0, :]  # Taking the [CLS] token's representation
        regression_output = self.activation(self.regression_head(pooled_output))

        return regression_output
