# Title Generator for Articles

This project contains code for a T5-small model fine-tuned for the task of generating summaries for articles.

## Project Content:

- **data_loader.py** - Loads the data.
- **data_preprocess.py** - Preprocesses (tokenizes) the data.
- **model_train.py** - Trains and evaluates the model.
- **model_evaluation.py** - Evaluates the model.
- **predict.py** - Runs the model on examples from the test set.

## Dataset for Training

Dataset
abisee/cnn_dailymail · Datasets at Hugging Face
The dataset used for the summarization task is called the CNN Dailymail dataset, it’s in English and contains over 300k unique news articles written by journalists at CNN and Daily Mail.
As for the data instances, for each instance there is a string for the article containing the body of the news article, a string for the highlights containing the highlight of the article written by the article author and one string for the id containing the heximal formatted SHA1 hash of the URL where the story was retrieved from. The average token count for the articles is 781 and for the highlights 56.


## Installation and Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/kriachers/ML_final_project.git
   cd ML_final_project
   cd SUMMARISATION
   cd summary_generator_model
   ```

2. **Model running:**

To run all model modules (data loading, data preprocessing, model training and evaluation) type in console:

```
python model_evaluation.py
```

3. **Generate summaries:**

After model is run and saved in following directory, use predict.py to generate titles for the test set examples:

```
python predict.py
```



