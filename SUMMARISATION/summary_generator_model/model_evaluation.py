from model_train import trainer
from data_preprocess import tokenized_small_cnn_dailymail

"""
MODEL EVALUATION
"""

results = trainer.predict(tokenized_small_cnn_dailymail["test"])
print(results)

"""
MODEL TEST
"""

test_results = trainer.evaluate(eval_dataset=tokenized_small_cnn_dailymail['test'])
print(test_results)