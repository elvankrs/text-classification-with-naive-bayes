# Text Classification with Naive Bayes

## Overview

This project implements multinomial and Bernoulli Naive Bayes algorithms for text classification on the [Reuters-21578](https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection) dataset. It covers data preprocessing, model training, evaluation, and statistical significance testing to compare the performance of different classifiers.

## Dataset

We use the [Reuters-21578](https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection) dataset, which contains 21,578 news articles classified into multiple topics. Only the top 10 topics are selected for this project, with a final vocabulary of 29,971 unique words.

## Evaluation

The models are evaluated using micro and macro F-scores on the development and test sets. Randomization tests are applied to assess the statistical significance of performance differences between the models.

## Results

Multinomial NB: Micro F1 = 0.932, Macro F1 = 0.664
Mutlivariate Bernoulli NB: Micro F1 = 0.851, Macro F1 = 0.499

The multinomial Naive Bayes model performs significantly better, as confirmed by the p-value (0.001) from the randomization test.

## Usage

- Python version: 3.9.13

To train and evaluate the multinomial Naive Bayes model:

```
python main.py data_path multinomial-nb
```

- `data_path` is the path to the dataset containing SGML files.  

To train and evaluate the multivariate Bernoulli Naive Bayes model:

```
python main.py data_path bernoulli-nb
```

For training both models with the best hyperparameters and running the randomization test:

```
python main.py data_path
```