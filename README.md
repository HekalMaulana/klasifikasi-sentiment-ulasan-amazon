# Sentiment Analysis on Amazon Product Reviews

## 📌 Overview
This project implements sentiment classification on Amazon product reviews based on the research paper *"PENGARUH KLASIFIKASI SENTIMEN PADA ULASAN PRODUK AMAZON BERBASIS REKAYASA FITUR DAN K-NEAREST NEIGHBOR"* by Nitami Lestari Putri et al. The system classifies reviews into positive, negative, or neutral sentiments using Feature Engineering and K-Nearest Neighbors (KNN) algorithm.

## 📂 Dataset
- **Source**: [Deception Detection on Amazon Reviews Dataset](https://github.com/aayush210789/Deception-Detection-on-Amazon-reviews-dataset)
- **Key Findings**:
  - 5 unique values in RATING column (1-5)
  - 2 unique values in VERIFIED_PURCHASE (Y/N)
  - No missing values in any columns

## 🔧 Features Used for Modeling
| Feature Type | Description |
|--------------|-------------|
| `RATING` | Product rating (1-5) |
| `VERIFIED_PURCHASE` | Purchase verification status (0/1) |
| `VOCAB` | 1400 most frequent words (TF-IDF vectors) |
| `LABEL` | Sentiment labels from TextBlob |

## 🛠️ Text Preprocessing Pipeline
1. Emoji removal
2. Case folding (lowercasing)
3. Punctuation removal
4. Tokenization
5. Stopword removal
6. Lemmatization (using WordNetLemmatizer)

## 🤖 Machine Learning Implementation
### Model Architecture
```python
Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1400)),
    ('knn', KNeighborsClassifier(
        n_neighbors=35,
        p=1.6,
        weights='distance'
    ))
])
