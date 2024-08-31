
# Sentiment Analysis for Coronavirus Comments

This project is aimed at performing sentiment analysis on Arabic comments related to the coronavirus pandemic. It classifies the comments into positive (1) and negative (0) sentiments.


## Requirements

- Python 3.x
- pandas
- numpy
- nltk
- scikit-learn
- demoji

## Setup

1. Clone the repository.
2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt

## Dataset

- **Training Data**: A CSV file (`train.csv`) containing Arabic comments labeled as positive (1) or negative (0).
- **Test Data**: A separate CSV file (`test.csv`) used for evaluating the model's performance.


## Data Preprocessing

Arabic text preprocessing is critical for effective sentiment analysis. This project includes several steps to clean and normalize the data:

1. **Stopword Removal**: Removes common Arabic stopwords using NLTK.
2. **Normalization**: Normalizes Arabic text (e.g., different forms of 'alif' to a standard form).
3. **Diacritic Removal**: Removes Arabic diacritics to reduce noise in the text.
4. **Number Removal**: Removes any numeric characters.
5. **Stemming**: Applies light stemming using ISRIStemmer.
6. **Punctuation Removal**: Removes punctuation to focus on the core text.
7. **Emoji Removal**: Removes emojis from the text using the `demoji` library.

## Feature Extraction

- **TF-IDF (Term Frequency-Inverse Document Frequency)**: This technique is used to convert text data into numerical features that can be fed into machine learning models. We use unigrams and bigrams to capture more context from the text.

## Model Training

Several machine learning models are explored for sentiment classification:

- **Random Forest Classifier**: A robust ensemble learning method for classification.
- **Support Vector Machine (SVM)**: A powerful classifier that works well with high-dimensional data.

## Hyperparameter Tuning

- **Grid Search with Cross-Validation**: The `GridSearchCV` method is used to find the best hyperparameters for the models. This process is crucial for optimizing model performance.
- **Random Forest Hyperparameters**: 
  - `n_estimators`: Number of trees in the forest.
  - `max_features`: Maximum number of features to consider for splitting.
  - `random_state`: Controls the randomness of the bootstrapping process.

## Evaluation and Results

- **Model Evaluation**: The models are evaluated based on accuracy, precision, recall, and F1-score using a confusion matrix.
- **Results**: The Random Forest model with tuned hyperparameters achieved the best performance.
  
