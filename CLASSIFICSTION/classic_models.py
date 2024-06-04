# %%
import pandas as pd
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize
from nltk.tag import pos_tag

import numpy as np

import gensim
import gensim.downloader as gensim
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

"""
FILE READING
"""
# %%
current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'clickbait_data.csv')

df = pd.read_csv(file_path)

"""
TEXT PREPROCESSING
"""
# %%
def preprocess_text(text):
  tokens = word_tokenize(text.lower())
  tokens = [token for token in tokens if token.isalnum()]
  return tokens

df['cleaned_headline'] = df['headline'].apply(preprocess_text)

X = df['cleaned_headline']
y = df['clickbait']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

vectorizer = TfidfVectorizer()
print("Training set size:", len(X_train))
print("Validation set size:", len(X_val))
print("Testing set size:", len(X_test))

"""
Feature extraction
"""
# %%
glove_model = gensim.load("glove-wiki-gigaword-50")

def feature_extraction(data):
    features_list = []
    for headline in data:
      features = {}
      # adding title length as feature since clickbait seems to have longer titles
      features["title_length"] = str(len(headline))
      # POS tags to determine writing style
      # nltk pos_tag also contains NER
      pos_tags = pos_tag(headline)
      features["pos_tags"] = " ".join(tag for _, tag in pos_tags)
      # average word length as feature
      avg_word_length = sum(len(word) for word in headline) / len(headline) if headline else 0
      features["average_word_length"] = str(avg_word_length)
      # adding word embeddings to provide further context
      word_embeddings = []
      for word in headline:
        try:
          word_embedding = glove_model[word]
        except KeyError:
          word_embedding = np.random.normal(size=50)
        word_embeddings.append(word_embedding)
      if word_embeddings:
        features["word_embeddings"] = " ".join(str(value) for value in np.mean(word_embeddings, axis = 0))
      else:
        features["word_embeddings"] = " ".join(str(value) for value in np.zeros(50))
      # everything needs to be in str format for the pipeline to work
      features_list.append(features)
    return features_list

"""
Model training and testing
"""

"""
Creating pipelines
"""
# %%
rf_pipeline = Pipeline([
    ("dict_vect", DictVectorizer()),  # Convert dictionaries to vectors
    ("rf", RandomForestClassifier(n_estimators=150, random_state=42))
])

dt_pipeline = Pipeline([
    ("dict_vect", DictVectorizer()),  # Convert dictionaries to vectors
    ("dt", DecisionTreeClassifier(random_state = 42, ccp_alpha = 1.25))
])

nb_pipeline = Pipeline([
    ("dict_vect", DictVectorizer()),  # Convert dictionaries to vectors
    ("nb",MultinomialNB(alpha = 1.5))
])

"""
Training and testing the different classic models
"""
# %%
def train_model(model, X_train, y_train, X_val, y_val):
  X_train_features = feature_extraction(X_train)
  X_val_features = feature_extraction(X_val)


  model.fit(X_train_features, y_train)

  y_val_pred = model.predict(X_val_features)

  accuracy = accuracy_score(y_val, y_val_pred)
  print("Validation Accuracy:", accuracy)

  report = classification_report(y_val, y_val_pred)
  print("Validation Classification Report: \n", report)

def test_model(model, X_test, y_test):
  X_test_features = feature_extraction(X_test)

  y_test_pred = model.predict(X_test_features)

  accuracy = accuracy_score(y_test, y_test_pred)
  print("Accuracy score:", accuracy)

  report = classification_report(y_test, y_test_pred)
  print("Validation Classification Report:\n", report)

print("Training Random Forest Classifier...")
print(train_model(rf_pipeline, X_train, y_train, X_val, y_val))

print("Testing Random Forest Classifier...")
print(test_model(rf_pipeline, X_test, y_test))

print("Training Decision Tree Classifier...")
print(train_model(dt_pipeline, X_train, y_train, X_val, y_val))

print("Training MultinomialNB Classifier...")
print(train_model(nb_pipeline, X_train, y_train, X_val, y_val))

print("Testing MultinomialNB Classifier...")
print(test_model(nb_pipeline, X_test, y_test))

"""
Testing best classic model on example sentences
"""
# %%
test_sentences = ["…they don’t want you to know", "dinosaurus are good","This Is How…","This is How Business Owners are Saving Thousands on Their Taxes","if disney princesses were from florida	"]
test = [preprocess_text(sentence) for sentence in test_sentences]

test_sentences_pred = nb_pipeline.predict(feature_extraction(test))
max_length = max(len("Original Sentence"), max(len(sentence) for sentence in test_sentences))

for tokens, sentence, label in zip(test, test_sentences, test_sentences_pred):
  print(f"Sentence: {sentence.ljust(max_length)} - Predicted Label: {'CLICKBAIT' if label == 1 else 'NOT CLICKBAIT'}")

"""
Testing best classic model on example sentences
"""
# %%
test_sentences = ["…they don’t want you to know", "dinosaurus are good","This Is How…","This is How Business Owners are Saving Thousands on Their Taxes","if disney princesses were from florida	"]
test = [preprocess_text(sentence) for sentence in test_sentences]

test_sentences_pred = nb_pipeline.predict(feature_extraction(test))
max_length = max(len("Original Sentence"), max(len(sentence) for sentence in test_sentences))

for tokens, sentence, label in zip(test, test_sentences, test_sentences_pred):
  print(f"Sentence: {sentence.ljust(max_length)} - Predicted Label: {'CLICKBAIT' if label == 1 else 'NOT CLICKBAIT'}")


"""
LIME
"""

"""
Lime with NB pipeline:
"""
# %%
from lime.lime_text import LimeTextExplainer

class_names = ['Not Clickbait', 'Clickbait']
explainer = LimeTextExplainer(class_names=class_names)

# Function to transform a single text input into the format expected by the pipeline
def transform_text(text):
    return [{"text": t} for t in text]

# Choose a document to explain
idx = 45

# Ensure the document is in string format
document_to_explain = " ".join(X_test.iloc[idx])

# Generate explanation
exp1 = explainer.explain_instance(
    document_to_explain,
    lambda x: nb_pipeline.predict_proba(transform_text(x)),
    num_features=6
)

print('Document id: %d' % idx)
print('Probability(Clickbait) =', nb_pipeline.predict_proba(transform_text([document_to_explain]))[0, 1])
print('True class: %s' % class_names[y_test.iloc[idx]])

exp1.as_list()

exp1.show_in_notebook(text=True)

fig1 = exp1.as_pyplot_figure()

"""
Lime with RF pipeline:
"""
# %%
class_names = ['Not Clickbait', 'Clickbait']
explainer = LimeTextExplainer(class_names=class_names)

# Function to transform a single text input into the format expected by the pipeline
def transform_text(text):
    return [{"text": t} for t in text]

# Choose a document to explain
idx = 34

# Ensure the document is in string format
document_to_explain = " ".join(X_test.iloc[idx])

# Generate explanation
exp2 = explainer.explain_instance(
    document_to_explain,
    lambda x: rf_pipeline.predict_proba(transform_text(x)),
    num_features=6
)

print('Document id: %d' % idx)
print('Probability(Clickbait) =', rf_pipeline.predict_proba(transform_text([document_to_explain]))[0, 1])
print('True class: %s' % class_names[y_test.iloc[idx]])

exp2.as_list()

exp2.show_in_notebook(text=True)

fig2 = exp2.as_pyplot_figure()


"""
Lime with DT pipeline:
"""
# %%
class_names = ['Not Clickbait', 'Clickbait']
explainer = LimeTextExplainer(class_names=class_names)

# Function to transform a single text input into the format expected by the pipeline
def transform_text(text):
    return [{"text": t} for t in text]

# Choose a document to explain
idx = 12

# Ensure the document is in string format
document_to_explain = " ".join(X_test.iloc[idx])

# Generate explanation
exp3 = explainer.explain_instance(
    document_to_explain,
    lambda x: dt_pipeline.predict_proba(transform_text(x)),
    num_features=6
)

print('Document id: %d' % idx)
print('Probability(Clickbait) =', dt_pipeline.predict_proba(transform_text([document_to_explain]))[0, 1])
print('True class: %s' % class_names[y_test.iloc[idx]])

exp3.as_list()

exp3.show_in_notebook(text=True)

fig3 = exp3.as_pyplot_figure()