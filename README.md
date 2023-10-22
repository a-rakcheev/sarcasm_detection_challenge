# Sarcasm detection challenge
### A one-day challenge to build model distinguishing real from sarcastic news headlines 

This challenge was part of the data science full-time bootcamp at [Constructor Academy in Zürich](https://academy.constructor.org/data-science/zurich).

## Data 
The data consists of ~8500 news headlines from HuffPost and The Onion (a satirical news outlet). It is labeled and the goal is to use NLP methods and supervised learning to get the best accuracy.
It seems that the original dataset stems from a [Kaggle competition](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection), however, it was provided from the bootcamp via Google drive and may have been modified at some point.

## Approach and code

- First the data was cleaned and preprocessed (**data_preprocessing.ipynb**)
- In the notebook (**train_with_validation.ipynb**) the data is split into train, validation, and test datasets. As a base model, ***logistic regression*** with ***bag-of-words*** embedding is set up; it achieves around **83% accuracy**. 
- Afterwards (in the same notebook) a ***BERT model** (['bert-base-uncased'](https://huggingface.co/bert-base-uncased)) with two additional dense and dropout layers is trained via tensorflow on a GPU. It achieves **~91 %** accuracy. Other BERT models and layer sizes have been tried (not included in notebook), but performed worse.
- Finally, the best model is retrained on the test and validation data in the notebook (**train_without_validation.ipynb**). Note that it is stopped manually after two epochs to prevent overfitting (based on observations from training with validation data).
- The final accuracy remains **~91%** but other metrics improve upon retraining (see the classification report).  
