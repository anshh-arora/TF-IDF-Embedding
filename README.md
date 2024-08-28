# TF-IDF Embedding Project

This repository contains scripts and a Jupyter notebook that demonstrate how to create TF-IDF (Term Frequency-Inverse Document Frequency) embeddings. The project showcases two approaches: using the `sklearn` library and building the TF-IDF algorithm from scratch.

## What is TF-IDF?

**TF-IDF** stands for Term Frequency-Inverse Document Frequency. It is a numerical statistic intended to reflect how important a word is to a document in a collection or corpus. The TF-IDF value increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus, helping to adjust for the fact that some words appear more frequently in general.

The formula for TF-IDF is:
\[
\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \text{IDF}(t)
\]
- **TF(t, d)**: Term Frequency of term `t` in document `d`.
- **IDF(t)**: Inverse Document Frequency of term `t` across all documents.

![TF-IDF Explanation](https://github.com/anshh-arora/TF-IDF-Embedding/blob/main/tf-idf.png)  

## Importance of TF-IDF in Language Models

TF-IDF is a fundamental technique used in natural language processing (NLP) and is particularly important in traditional information retrieval systems. It helps in identifying the most relevant words in a document, which is crucial for search engines, document classification, and text mining.

In the context of large language models (LLMs), TF-IDF is often used as a feature engineering step before applying more complex models. It provides a strong baseline by converting text into numerical features that reflect the importance of words in the document, which can significantly improve the performance of machine learning models.

## Project Structure

This repository contains three main files:

1. **`sklearn_TD-IDF.py`**: A Python script that creates TF-IDF embeddings using the `sklearn` library.
2. **`TF-IDF_from_scratch.py`**: A Python script that implements TF-IDF from scratch, without relying on external libraries.
3. **`TF_IDF From Scratch.ipynb`**: A Jupyter notebook that walks through the creation of TF-IDF embeddings from scratch, with detailed explanations and visualizations.

## How to Clone the Repository

To clone this repository, use the following command:

1. ```bash
   git clone https://github.com/anshh-arora/TF-IDF-Embedding
   cd tfidf-embedding-project

## Contact Information
For any questions or feedback, feel free to reach out:

- **Email**: [ansharora.cs@gmail.com](mailto:ansharora.cs@gmail.com)
- **LinkedIn**: [Connect with me on LinkedIn](https://www.linkedin.com/in/ansh-arora-data-scientist/)
- **Kaggle**: [Follow me on Kaggle](https://www.kaggle.com/ansh1529)

