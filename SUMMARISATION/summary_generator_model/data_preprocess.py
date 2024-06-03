from transformers import T5Tokenizer
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from data_loader import small_cnn_dailymail

checkpoint_small = "t5-small"
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def preprocess_function(item):

  labels = tokenizer(text=item["highlights"], max_length=56, truncation=True)
  inputs = tokenizer(text=item["article"], max_length=400, truncation=True)

  model_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
  model_inputs["labels"] = labels["input_ids"]
  return model_inputs

tokenized_small_cnn_dailymail = small_cnn_dailymail.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint_small)