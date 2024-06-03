# Sentiment Analysis Research Paper

## Introduction
This repository contains the code and resources for our research paper on **Sentiment Analysis**. Sentiment analysis, also known as opinion mining, is a natural language processing (NLP) technique used to determine whether data is positive, negative, or neutral. This research focuses on analyzing sentiment from various data sources using advanced machine learning algorithms.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

## Dataset
We used a combination of publicly available sentiment datasets

Each dataset contains labeled data that indicates whether the sentiment of a given text is positive, negative, or neutral.

## Methodology
Our methodology consists of the following steps:
1. **Data Preprocessing**: Cleaning and preparing the data for analysis.
2. **Feature Extraction**: Extracting relevant features from the text using techniques such as TF-IDF, word embeddings, etc.
3. **Model Training**: Training various machine learning models including Logistic Regression, SVM, and Neural Networks.
4. **Evaluation**: Evaluating the performance of the models using metrics such as accuracy, precision, recall, and F1-score.

## Results
Our analysis yielded the following key findings:
- Random Forest Classifier: Achieved an accuracy of 84.92% with a precision of 83% and recall of 89%.
- Naive Bayes Theorem: Achieved an accuracy of 82.01% with a precision of 85% and recall of 78%.
- SVM : Achieved an accuracy of 85.98% with a precision of 87% and recall of 85%.

Detailed results and visualizations can be found in the RESULTS directory.

## Conclusion
Our research demonstrates that Financial Market is greatky affected by the sentiment of news .

## Installation
To replicate our results, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/jheelamH/sentiment-analysis-research
    cd sentiment-analysis-research
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the sentiment analysis script, use the following command:
```bash
python sentiment_analysis.py --input data/input_file.txt --output results/output_file.txt
