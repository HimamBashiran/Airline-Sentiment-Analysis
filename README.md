# Airline Twitter Sentiment Analysis

## Description

This project investigates public sentiment toward airline services by analyzing Twitter posts using Natural Language Processing (NLP) and classical machine learning approaches. The goal is to automatically classify tweets into **positive** or **negative** sentiment classes based on textual content.

The analysis leverages a real-world social media dataset, applying systematic text preprocessing, feature extraction via TF-IDF, and multiple supervised learning algorithms. This project demonstrates an end-to-end sentiment analysis pipeline, suitable for academic study, applied NLP practice, and baseline comparison before deploying more advanced deep learning or transformer-based models.

---

## Methods

### Dataset Preparation

The dataset is obtained from the CrowdFlower Airline Twitter Sentiment dataset. Tweets labeled as *neutral* are removed to simplify the task into binary sentiment classification (positive vs. negative).

### Exploratory Data Analysis (EDA)

* Inspection of dataset structure and missing values
* Visualization of sentiment distribution using bar and pie charts
* Identification of class imbalance

### Text Preprocessing

To improve data quality and model performance, the following preprocessing steps are applied:

1. Removal of Twitter-specific noise (mentions, hashtags, URLs, retweet markers)
2. Lowercasing and punctuation removal
3. Stopword removal using NLTK English stopwords
4. Duplicate tweet elimination
5. Stemming using Porter Stemmer
6. Lemmatization using spaCy (`en_core_web_sm`)

### Feature Extraction

Text data is transformed into numerical vectors using **Term Frequency–Inverse Document Frequency (TF-IDF)**. This method captures the importance of words relative to the entire corpus while reducing the impact of frequently occurring but less informative terms.

### Model Training

The dataset is split into training and testing sets using an 80:20 ratio. The following models are trained:

* Baseline classifier (majority class prediction)
* Logistic Regression
* Random Forest Classifier
* K-Nearest Neighbors (KNN)

### Evaluation

Model performance is evaluated using:

* Accuracy
* Precision (weighted)
* Recall (weighted)
* F1-score (weighted)
* Confusion matrices for error analysis

---

## Results

### Model Performance Comparison

| Model               | Accuracy | Precision | Recall   | F1-Score |
| ------------------- | -------- | --------- | -------- | -------- |
| Random Forest       | 0.893852 | 0.888520  | 0.893852 | 0.887722 |
| Logistic Regression | 0.885007 | 0.881674  | 0.885007 | 0.872343 |
| K-Nearest Neighbors | 0.730650 | 0.733696  | 0.730650 | 0.732152 |

The table above shows that **Random Forest** achieves the highest performance across all evaluation metrics, followed closely by **Logistic Regression**. **KNN** demonstrates substantially lower performance, indicating its limitations when applied to high-dimensional sparse TF-IDF features.

### Discussion

Overall, **Random Forest emerges as the best-performing model** in this study, followed closely by Logistic Regression. The results confirm that ensemble and linear models are well-suited for TF-IDF–based sentiment classification, while distance-based methods like KNN are more sensitive to feature dimensionality and data sparsity.

---

## Future Improvements

* Handle class imbalance using resampling (SMOTE, undersampling)
* Use advanced embeddings (Word2Vec, GloVe, FastText)
* Apply transformer-based models (BERT, RoBERTa)
* Extend to multi-class sentiment classification
* Hyperparameter tuning and cross-validation

---

## References

1. CrowdFlower. *Airline Twitter Sentiment Dataset*. Available at: [https://data.world/crowdflower/airline-twitter-sentiment](https://data.world/crowdflower/airline-twitter-sentiment)
2. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
3. Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed.). Draft.
4. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
5. Hutto, C. J., & Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. *ICWSM*.

---

## Author

**Himam Bashiran**
Data Science Student
