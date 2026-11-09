# Maganor AI: Exploring the Challenges of Detecting Harmful Language Online

## Project Overview

Maganor AI is an exploratory research project investigating how machine learning and transformer-based language models identify toxic and harmful language in online comments.

The project focuses not only on model performance but also on understanding why automated systems sometimes misclassify comments, particularly when language involves indirect hostility, ambiguity, sarcasm, or contextual meaning.

The central motivation is to explore the limitations of automated language understanding and the importance of developing more responsible AI systems for online environments.

## Research Question

How effectively can pretrained language models identify toxic and harmful online comments, and what limitations do they demonstrate when interpreting context, sarcasm, indirect hostility, and gender-based harassment?

## Dataset

The experiments use the Jigsaw Toxic Comment Classification dataset, containing 159,571 English-language comments from Wikipedia discussion pages.

The dataset includes six toxicity-related labels:

- Toxic
- Severe toxic
- Obscene
- Threat
- Insult
- Identity hate

For the main experiments, the task was simplified into binary classification: toxic and non-toxic.

## Methodology

Three classification experiments were conducted:

1. **TF-IDF + Logistic Regression:** A traditional machine learning baseline.
2. **DistilBERT (128 tokens):** A pretrained transformer fine-tuned for binary toxic comment classification.
3. **DistilBERT (256 tokens):** A second experiment investigating whether a longer input length improves classification performance.

The models were evaluated using accuracy, precision, recall, F1-score, confusion matrices, and prediction error analysis.

## Experimental Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 94.04% | 64.07% | 86.04% | 73.45% |
| DistilBERT (128 tokens) | 96.68% | 84.95% | 79.37% | 82.07% |
| DistilBERT (256 tokens) | 96.64% | 84.76% | 79.24% | 81.91% |

DistilBERT with 128 tokens achieved the highest accuracy, precision, and F1-score among the three experiments. Logistic Regression achieved the highest recall.

Further analysis revealed an association between longer comments and false-negative predictions in both DistilBERT experiments.

## Key Findings

- DistilBERT achieved stronger precision and F1-score than the traditional baseline in these experiments.
- Higher overall accuracy does not guarantee that every harmful comment will be identified.
- Longer comments were associated with higher false-negative rates.
- Increasing the input length from 128 to 256 tokens did not improve the overall evaluation metrics.
- Error analysis is essential for understanding the limitations of automated content classification.

## Limitations

This project is an exploratory research prototype, not a production-ready moderation system.

The dataset consists of English Wikipedia comments and does not represent Ethiopian social-media environments or local languages. The experiments also used limited training samples and computational resources.

The classifier predicts dataset toxicity labels. It does