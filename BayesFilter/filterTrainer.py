from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd
import joblib
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Download NLTK resources for text processing
nltk.download('wordnet')  # Required for lemmatization
nltk.download('stopwords')  # Required for filtering out common stop words

# Initialize the lemmatizer and stop words set
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text: str) -> str:
    """
    Preprocesses the input text for machine learning.

    This function performs the following operations:
    - Converts text to lowercase
    - Removes special characters and numbers
    - Tokenizes the text into words
    - Lemmatizes each word to its base form
    - Filters out stop words

    Parameters:
    text (str): The input text to preprocess.

    Returns:
    str: The cleaned and processed text.

    NOTE: This function assumes that the input text is a single string. 
    Edge cases such as empty strings or strings with only stop words will return an empty string.
    """
    # Lowercase and remove special characters/numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower().strip())
    # Tokenize and lemmatize, filtering out stop words
    words = text.split()
    return ' '.join([lemmatizer.lemmatize(word) for word in words if word not in stop_words])

# Load dataset from CSV file
df = pd.read_csv('coral_preservation_dataset.csv')

# Check class balance in the dataset
print("Class distribution:\n", df['label'].value_counts())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
)

# Create an enhanced machine learning pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        preprocessor=preprocess,  # Use the custom preprocessing function
        ngram_range=(1, 2),  # Include unigrams and bigrams
        max_df=0.75,         # Ignore terms that appear in more than 75% of documents
        min_df=2             # Ignore terms that appear in less than 2 documents
    )),
    ('classifier', RandomForestClassifier(
        class_weight='balanced',  # Adjust weights to handle class imbalance
        n_jobs=-1  # Use all available cores for training
    ))
])

# Define hyperparameter grid for model tuning
params = {
    'tfidf__max_features': [5000, 10000],  # Maximum number of features to consider
    'classifier__n_estimators': [100, 200],  # Number of trees in the forest
    'classifier__max_depth': [None, 50]  # Maximum depth of the tree
}

# Perform grid search with cross-validation to find the best model parameters
model = GridSearchCV(pipeline, params, cv=3, scoring='f1_weighted', verbose=1)
model.fit(X_train, y_train)  # Fit the model to the training data

# Output the best parameters found during grid search
print("Best parameters:", model.best_params_)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance using a classification report
print(classification_report(y_test, y_pred))

# Save the best model to a file for future use
joblib.dump(model.best_estimator_, 'coral_classifier.joblib')