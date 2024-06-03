import datasets
from datasets import load_dataset, DatasetDict

"""
FILE READING
"""

cnn_dailymail = load_dataset("abisee/cnn_dailymail", "3.0.0")

small_cnn_dailymail = DatasetDict(
    train = cnn_dailymail["train"].shuffle(seed=24).select(range(800)),
    validation= cnn_dailymail["validation"].shuffle(seed=24).select(range(300)),
    test = cnn_dailymail["test"].shuffle(seed=24).select(range(150))
)


