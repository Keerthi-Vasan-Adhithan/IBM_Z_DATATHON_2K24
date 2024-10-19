# IBM_Z_DATATHON_2K24

## Algorithm:
1) Load the dataset using pandas and handle parsing errors if needed.
2) Inspect the dataset by printing the column names to identify the tweet text and target label columns.
3) preprocess the tweets by defining a function to clean them (removing URLs, mentions, hashtags, punctuation, converting to lowercase, tokenizing, removing stopwords, and lemmatizing).
4) apply the preprocessing function to the tweet column and store the cleaned text in a new column.
5) prepare the data for model training by setting the cleaned tweets as input features (X) and the labels as the target (y). Split into training and testing sets (80/20).
6) convert the text to numerical features using the TF-IDF vectorizer, fitting it on the training data and transforming both training and test sets.
7) train a Logistic Regression model using the TF-IDF-transformed data.
8) define a function to preprocess, transform, and predict suicide probability for new tweets.
9) evaluate the model by testing it on the test set, computing accuracy, and generating a classification report.
10) test with a sample tweet by passing it to the prediction function and printing the probability of a suicide attempt.

## Complete Dataset:
https://www.kaggle.com/datasets/kazanova/sentiment140

## Installation:
```py
pip install pandas numpy scikit-learn nltk streamlit
```

## Program:
```py
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load dataset
file_path = 'first.csv'  # Replace with the actual path to your dataset
data = pd.read_csv(file_path)

# Extract the text column (assuming it's the last column)
data['text'] = data.iloc[:, -1]

# Text cleaning function
def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove mentions (@user)
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove punctuation and numbers
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Apply the cleaning function to the text data
data['cleaned_text'] = data['text'].apply(clean_text)

# Define keywords for each risk level
sad_keywords = ['sad', 'unhappy', 'down', 'upset', 'disappointed']
moderate_keywords = ['anxiety', 'stress', 'nervous', 'worried', 'fear', 'tired', 'burnout']
high_risk_keywords = ['depression', 'suicide', 'hopeless', 'self-harm', 'lonely', 'mental health', 'therapy']

# Function to label the mental health risk status
def label_risk_status(text):
    for keyword in high_risk_keywords:
        if keyword in text:
            return 'High Risk'
    for keyword in moderate_keywords:
        if keyword in text:
            return 'Moderate'
    for keyword in sad_keywords:
        if keyword in text:
            return 'Sad'
    return 'Neutral'  # If no keyword is matched

# Apply the labeling function to the cleaned text
data['risk_status'] = data['cleaned_text'].apply(label_risk_status)

# Filter out "Neutral" rows
filtered_data = data[data['risk_status'] != 'Neutral']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(filtered_data['cleaned_text'], filtered_data['risk_status'], test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Build and train the Logistic Regression model for multi-class classification
multi_class_model = LogisticRegression(multi_class='ovr')  # one-vs-rest strategy for multi-class classification
multi_class_model.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = multi_class_model.predict(X_test_tfidf)


# Get the count of each risk status in the test set
risk_status_counts = y_test.value_counts()

# Display the result showing how many are classified as Sad, Moderate, or High Risk
print("Risk Status Counts:")
print(risk_status_counts)

# Filter the dataset to show only the rows classified as "High Risk"
high_risk_records = filtered_data[filtered_data['risk_status'] == 'High Risk']

# Display the high risk records with all details
print("High Risk Records:")
print(high_risk_records)
```
## Output:
![WhatsApp Image 2024-10-20 at 00 34 19_e954e4fb](https://github.com/user-attachments/assets/76a3a5b4-3ad8-44b3-881c-084069f22ef1)

![WhatsApp Image 2024-10-20 at 00 34 34_ed3662c4](https://github.com/user-attachments/assets/7099e649-2342-4df2-8545-d5039131971c)

