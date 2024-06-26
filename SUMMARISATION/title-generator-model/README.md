# Title Generator for Articles

This project contains code for a T5-small model fine-tuned for the task of generating titles for articles.

## Project Content:

- **data_loader.py** - Loads the data.
- **data_preprocess.py** - Preprocesses (tokenizes) the data.
- **model_train.py** - Trains and evaluates the model.
- **predict.py** - Runs the model on examples from the test set.

## Dataset for Training

The training used a truncated version of the Medium Articles Dataset. The original dataset, which consists of 150,000 rows, was truncated to 20,000 rows and saved as `small_medium_articles.csv`.

## Installation and Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/kriachers/ML_final_project.git
   cd ML_final_project
   cd SUMMARISATION
   cd title-generator-model
   ```

2. **Install the required packages:**

```
pip install -r requirements.txt
```

OR type following commands in the terminal 

```
python -m pip install pandas
python -m pip install numpy
python -m pip install transformers
python -m pip install datasets
python -m pip install sentencepiece
python -m pip install evaluate
python -m pip install rouge_score
python -m pip install torch
python -m pip install accelerate
python -m pip install protobuf
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



