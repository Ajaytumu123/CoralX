# filter.py
import joblib

# MUST HAVE THE SAME PREPROCESSING FUNCTION
def preprocess(text: str) -> str:
    """
    Preprocesses the input text for model prediction.

    This function converts the input text to lowercase and strips any leading or trailing whitespace.
    
    Parameters:
    - text (str): The input text to be preprocessed.

    Returns:
    - str: The preprocessed text, ready for model input.
    
    NOTE: This function must match the preprocessing used during model training.
    """
    return text.lower().strip()

# Load the trained model
try:
    model = joblib.load('coral_classifier.joblib')
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit()

def bayes_filter(query: str, irrelevant_threshold: float = 0.7, min_confidence: float = 0.6) -> tuple:
    """
    Applies a Bayesian filter to determine if a query is relevant or irrelevant.

    This function predicts the probability of the query belonging to each class and checks
    if it should be allowed based on the predicted class and confidence levels.

    Parameters:
    - query (str): The input query to be evaluated.
    - irrelevant_threshold (float): The threshold above which a query is considered irrelevant.
    - min_confidence (float): The minimum confidence level required for a query to be considered relevant.

    Returns:
    - tuple: A tuple containing:
        - bool: True if the query is allowed, False if blocked.
        - float: The confidence level of the prediction.

    Potential Side Effects:
    - May print error messages if prediction fails.

    Notable Edge Cases:
    - If the model fails to predict, it defaults to allowing the query through (fail-safe).
    """
    try:
        proba = model.predict_proba([query])[0]  # Get predicted probabilities for each class
        confidence = proba.max()  # Maximum confidence level from the probabilities
        predicted_class = model.classes_[proba.argmax()]  # Class with the highest probability
        
        # Block if either:
        # 1. Predicted irrelevant with high confidence
        # 2. Overall confidence is too low (uncertain prediction)
        if (predicted_class == "irrelevant" and confidence > irrelevant_threshold) \
            or (confidence < min_confidence):
            return False, confidence  # Block the query
        return True, confidence  # Allow the query
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return True, 0.0  # Fail-safe: allow query through

# Example usage
if __name__ == "__main__":
    # List of test queries to evaluate the bayes_filter function
    test_queries = [
        "What is the tastiest fish?",
        "how can we feed food to sea horse",
        "How to restore damaged coral reefs?",
        "Write Python code for image recognition",
        "What pH level is best for coral growth?",
        "How to make seafood pasta?",
        "Coral bleaching prevention methods",
        "Latest football match results",
        "Deep learning architecture for marine biology",
        "Best snorkeling spots near reefs",
        "How do corals reproduce?",
        "JavaScript framework comparison",
        "Impact of microplastics on corals",
        "Chicken tikka masala recipe",
        "Coral disease identification guide",
        "Machine learning hyperparameter tuning",
        "Ocean temperature monitoring techniques",
        "How to train a neural network",
        "Symbiotic relationships in coral reefs",
        "Car maintenance tips",
        "Marine protected areas management",
        "Quantum computing basics"
    ]
    
    # Evaluate each query using the bayes_filter function
    for query in test_queries:
        allowed, confidence = bayes_filter(query)
        print(f"Query: {query}")
        print(f"Allowed: {allowed} | Confidence: {confidence:.2f}")
        print("-" * 50)