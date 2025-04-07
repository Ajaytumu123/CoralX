import csv
import time
import requests
import json
from tqdm import tqdm
import random
import os  # Added import for file existence check

# Configuration
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"  # API endpoint for question generation
MODEL_NAME = "qwen2.5-coder:3b"  # Model name used for generating questions
DATASET_SIZE = 20000  # Total number of questions to generate
CSV_FILENAME = "coral_preservation_dataset.csv"  # Output CSV file name
BATCH_SIZE = 10  # Number of questions to generate in each API call
LABELS = ['relevant', 'irrelevant'] * (DATASET_SIZE // 2)  # Balanced labels for dataset
random.shuffle(LABELS)  # Randomize label order upfront to ensure diversity

def validate_response(text):
    """
    Ensure response is clean and properly formatted.

    Parameters:
    - text (str): The raw response text from the API.

    Returns:
    - str: A cleaned version of the response text, stripped of unwanted characters.

    Side Effects:
    - None

    Notable Edge Cases:
    - If the input text is empty or malformed, the function will still return an empty string.
    """
    return text.strip().replace('"', '').replace('\r', '').replace('\\', '')

def generate_batch(label):
    """
    Generate a batch of questions based on the specified label.

    Parameters:
    - label (str): The category of questions to generate ('relevant' or 'irrelevant').

    Returns:
    - list: A list of generated questions in JSON format, or None if the generation fails.

    Side Effects:
    - Makes an API call to the OLLAMA endpoint.

    Notable Edge Cases:
    - If the API response is not valid JSON, the function will retry until a valid response is received.
    """
    system_prompt = f"""Generate {BATCH_SIZE} questions about {label} topics. Follow EXACTLY this JSON format:
    {{
        "questions": [
            "question 1",
            "question 2",
            ...
        ]
    }}
    {'Focus on coral preservation, marine biology' if label == 'relevant' else 'NO marine topics - choose diverse subjects'}"""
    
    try:
        response = requests.post(
            OLLAMA_ENDPOINT,
            json={
                "model": MODEL_NAME,
                "prompt": system_prompt,
                "system": "You are a precise data generator. Return ONLY valid JSON.",
                "format": "json",
                "stream": False
            }
        )
        
        if response.status_code == 200:
            try:
                data = json.loads(response.json()["response"])  # Parse the JSON response
                return [validate_response(q) for q in data.get("questions", [])]  # Clean and return questions
            except json.JSONDecodeError:
                print("Invalid JSON response, retrying...")  # Log error and retry
                return None
        return None  # Return None if the response status is not 200
    except Exception as e:
        print(f"API Error: {str(e)}")  # Log any exceptions that occur during the API call
        return None

def main():
    """
    Main function to generate a dataset of questions and save them to a CSV file.

    Parameters:
    - None

    Returns:
    - None

    Side Effects:
    - Writes generated questions to a CSV file.

    Notable Edge Cases:
    - If the CSV file already exists and is empty, it will still write the header.
    - The function will handle retries for generating valid batches of questions.
    """
    # Check if file exists and if it's empty
    file_exists = os.path.isfile(CSV_FILENAME)  # Check for file existence
    file_empty = False
    if file_exists:
        file_empty = os.path.getsize(CSV_FILENAME) == 0  # Check if the file is empty

    # Determine file open mode
    mode = 'a' if file_exists else 'w'  # Append if file exists, otherwise write new

    with open(CSV_FILENAME, mode, newline='', encoding='utf-8') as file:
        writer = csv.writer(file)  # Create a CSV writer object
        
        # Write header only if file is new or empty
        if not file_exists or file_empty:
            writer.writerow(['text', 'label'])  # Write header for CSV

        progress = tqdm(total=DATASET_SIZE)  # Initialize progress bar
        generated_count = 0  # Counter for generated questions
        
        while generated_count < DATASET_SIZE:  # Loop until the desired dataset size is reached
            current_label = LABELS[generated_count % len(LABELS)]  # Select label based on count
            
            batch = None
            while not batch:  # Retry until valid response
                batch = generate_batch(current_label)  # Generate a batch of questions
                if not batch:
                    time.sleep(1)  # Wait before retrying
            
            # Write immediately to prevent data loss
            writer.writerows([(question, current_label) for question in batch])  # Write questions to CSV
            file.flush()  # Force write to disk
            
            generated_count += len(batch)  # Update the count of generated questions
            progress.update(len(batch))  # Update progress bar
            
            time.sleep(0.3)  # Adjusted rate limiting to avoid overwhelming the API

if __name__ == "__main__":
    main()  # Execute the main function when the script is run