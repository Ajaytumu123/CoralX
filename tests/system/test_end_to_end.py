import os
import sys
import unittest
from unittest.mock import patch
import pytest
import subprocess

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class TestEndToEnd(unittest.TestCase):
    """End-to-end tests for the coral research system"""
    
    def setUp(self):
        """Set up test environment"""
        # Store the path to the scripts directory
        self.script_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
            'scripts', 'Hybrid-scripts', 'item_05_chain_retriever.py'
        )
        
        # Ensure log directory exists
        self.log_dir = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
            'logs'
        )
        os.makedirs(self.log_dir, exist_ok=True)
    
    @pytest.mark.slow
    @patch('builtins.input', return_value='yes')
    def test_coral_query_full_system(self, mock_input):
        """
        Test the system with a coral-related query
        
        This test requires the full system setup including:
        - Available model file
        - Vector database
        - OpenAI API key
        
        Note: This test is marked as 'slow' and should be run selectively.
        """
        # Run the command with a relevant query that should pass the filter
        query = "What causes coral bleaching?"
        
        try:
            # Create a temporary Python file to run the test
            temp_file = os.path.join(self.log_dir, 'temp_test.py')
            with open(temp_file, 'w') as f:
                f.write(f'''
import os
import sys
sys.path.insert(0, "{os.path.dirname(self.script_path)}")
from item_05_chain_retriever import answer_to_QA

result = answer_to_QA("{query}")
print("TEST_OUTPUT_MARKER")
print(result)
print("TEST_OUTPUT_END_MARKER")
''')
            
            # Run the script
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=60  # Timeout after 60 seconds
            )
            
            # Check for expected output patterns
            output = result.stdout
            
            # Extract just the result part
            if "TEST_OUTPUT_MARKER" in output and "TEST_OUTPUT_END_MARKER" in output:
                start_idx = output.find("TEST_OUTPUT_MARKER") + len("TEST_OUTPUT_MARKER")
                end_idx = output.find("TEST_OUTPUT_END_MARKER")
                result_output = output[start_idx:end_idx].strip()
                
                # Check for expected patterns in the output
                self.assertIn("FINAL ANSWER", result_output)
                
                # The answer should contain coral-related terms
                coral_terms = ["coral", "reef", "bleaching", "ocean", "temperature", "climate"]
                has_coral_term = any(term in result_output.lower() for term in coral_terms)
                self.assertTrue(has_coral_term, "Response should contain coral-related terminology")
                
                # Check for sources
                self.assertIn("Sources:", result_output)
            else:
                self.fail("Could not find test output markers in subprocess output")
            
        except subprocess.TimeoutExpired:
            self.fail("Test timed out - possibly due to hanging on API calls or user input")
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    @pytest.mark.slow
    @patch('builtins.input', return_value='no')
    def test_irrelevant_query_rejection(self, mock_input):
        """Test the system with an irrelevant query that should be rejected"""
        # This is an obviously irrelevant query
        query = "Best recipe for chocolate cake"
        
        try:
            # Create a temporary Python file to run the test
            temp_file = os.path.join(self.log_dir, 'temp_test.py')
            with open(temp_file, 'w') as f:
                f.write(f'''
import os
import sys
sys.path.insert(0, "{os.path.dirname(self.script_path)}")
from item_05_chain_retriever import answer_to_QA

# Override input to simulate rejection
import builtins
original_input = builtins.input
builtins.input = lambda *args: "no"

try:
    result = answer_to_QA("{query}")
    print("TEST_OUTPUT_MARKER")
    print(result)
    print("TEST_OUTPUT_END_MARKER")
finally:
    # Restore input function
    builtins.input = original_input
''')
            
            # Run the script
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=60  # Timeout after 60 seconds
            )
            
            # Check for expected output patterns
            output = result.stdout
            
            # Extract just the result part
            if "TEST_OUTPUT_MARKER" in output and "TEST_OUTPUT_END_MARKER" in output:
                start_idx = output.find("TEST_OUTPUT_MARKER") + len("TEST_OUTPUT_MARKER")
                end_idx = output.find("TEST_OUTPUT_END_MARKER")
                result_output = output[start_idx:end_idx].strip()
                
                # Check for expected patterns in the output
                self.assertIn("QUERY CANCELLED", result_output)
                self.assertIn("CoralX Foundation", result_output)
                self.assertNotIn("FINAL ANSWER", result_output)
            else:
                self.fail("Could not find test output markers in subprocess output")
            
        except subprocess.TimeoutExpired:
            self.fail("Test timed out - possibly due to hanging on API calls or user input")
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == '__main__':
    unittest.main() 