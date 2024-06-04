
# %%
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments
from transformers import Trainer
import pandas as pd
import os
import torch
from transformers import pipeline
from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset
import numpy as np
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification

"""
FILE READING
"""

# %%
output_dir = "clickbait_classifier"
os.makedirs(output_dir, exist_ok=True)

current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'clickbait_data.csv')

df = pd.read_csv(file_path)

"""
TOKENIZING
"""
# %%
df['headline'] = df['headline'].str.lower()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

"""
DATA SPLITTING
"""
# %%
hf_dataset = Dataset.from_pandas(df)

def truncate(example):
    return {
        'headline': " ".join(example['headline'].split()[:50]),
        'label': example['clickbait']
    }

small_dataset = DatasetDict(
    train=hf_dataset.shuffle(seed=24).select(range(800)).map(truncate),
    val=hf_dataset.shuffle(seed=24).select(range(800, 900)).map(truncate),
    test=hf_dataset.shuffle(seed=24).select(range(900, 1000)).map(truncate),
)

def tokenize_function(examples):
    return tokenizer(examples["headline"], padding=True, truncation=True)

small_tokenized_dataset = small_dataset.map(tokenize_function, batched=True, batch_size=16)

"""
MODEL TRAINING
"""
# %%

model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=2)

arguments = TrainingArguments(
    output_dir= output_dir,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    load_best_model_at_end=True,
    seed=224
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": np.mean(predictions == labels)}

def compute_loss(model, inputs):
    outputs = model(**inputs)
    logits = outputs.logits
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, model.config.num_labels), inputs["label"].view(-1))
    return loss

trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=small_tokenized_dataset['train'],
    eval_dataset=small_tokenized_dataset['val'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()

"""
MODEL SAVE
"""
# %%
trainer.save_model(output_dir)

"""
MODEL EVALUATE on eval_dataset parameter
"""
# %%
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

"""
TEST
"""
# %%
distilbert_results = trainer.predict(small_tokenized_dataset['test'])
print(distilbert_results)

