# Title Generator for Articles

This project contains code for a T5-small model fine-tuned for the task of generating summaries for articles.

## Project Content:

- **data_loader.py** - Loads the data.
- **data_preprocess.py** - Preprocesses (tokenizes) the data.
- **train.py** - Trains and evaluates the model.
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

2. **Install the required packages:**

```
pip install -r requirements.txt
```
3. **Data Loading:**

Use data_loader.py to load the dataset:
```
python data_loader.py
```

4. **Data Preprocessing:**
Preprocess the dataset using data_preprocess.py:

```
python data_preprocess.py
```

5. **Model Training and Evaluation:**


Train and evaluate the model using model_train.py:

```
python model_train.py
```

5. **Generate Titles::**


Use predict.py to generate titles for the test set examples:

```
python predict.py
```



