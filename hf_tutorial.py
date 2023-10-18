from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    training_args = TrainingArguments(output_dir="test_trainer")
    metric = evaluate.load("accuracy")
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()