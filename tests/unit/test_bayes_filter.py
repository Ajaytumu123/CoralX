import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import joblib
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the functions to test
from scripts.Hybrid_scripts.item_05_chain_retriever import preprocess, bayes_filter

class TestBayesFilter(unittest.TestCase):
    """Unit tests for the Bayes filter functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock model
        self.mock_model = MagicMock()
        self.mock_model.classes_ = ["relevant", "irrelevant"]
    
    def test_preprocess(self):
        """Test the preprocess function"""
        # Test basic preprocessing
        self.assertEqual(preprocess("Test Query"), "test query")
        self.assertEqual(preprocess(" TRIM SPACES "), "trim spaces")
        self.assertEqual(preprocess(""), "")
    
    @patch('scripts.Hybrid_scripts.item_05_chain_retriever.model', None)
    def test_bayes_filter_no_model(self):
        """Test filter behavior when model is not available"""
        is_allowed, confidence = bayes_filter("test query")
        self.assertTrue(is_allowed)  # Should allow all queries when model is not available
        self.assertEqual(confidence, 0.0)  # Confidence should be 0
    
    @patch('scripts.Hybrid_scripts.item_05_chain_retriever.model')
    def test_bayes_filter_relevant_query(self, mock_model):
        """Test filter behavior with a relevant query"""
        # Configure the mock model to predict relevant
        mock_model.predict_proba.return_value = [[0.2, 0.8]]  # 80% relevant, 20% irrelevant
        mock_model.classes_ = ["irrelevant", "relevant"]
        
        is_allowed, confidence = bayes_filter("coral bleaching causes")
        
        self.assertTrue(is_allowed)  # Should allow relevant queries
        self.assertEqual(confidence, 0.8)  # Confidence should be 0.8
        
    @patch('scripts.Hybrid_scripts.item_05_chain_retriever.model')
    def test_bayes_filter_irrelevant_query(self, mock_model):
        """Test filter behavior with an irrelevant query"""
        # Configure the mock model to predict irrelevant with high confidence
        mock_model.predict_proba.return_value = [[0.9, 0.1]]  # 90% irrelevant, 10% relevant
        mock_model.classes_ = ["irrelevant", "relevant"]
        
        is_allowed, confidence = bayes_filter("recipe for chicken soup")
        
        self.assertFalse(is_allowed)  # Should block irrelevant queries
        self.assertEqual(confidence, 0.9)  # Confidence should be 0.9
    
    @patch('scripts.Hybrid_scripts.item_05_chain_retriever.model')
    def test_bayes_filter_low_confidence(self, mock_model):
        """Test filter behavior with a low confidence prediction"""
        # Configure the mock model to predict with low confidence
        mock_model.predict_proba.return_value = [[0.55, 0.45]]  # 55% irrelevant, 45% relevant
        mock_model.classes_ = ["irrelevant", "relevant"]
        
        is_allowed, confidence = bayes_filter("ocean temperatures", min_confidence=0.6)
        
        self.assertFalse(is_allowed)  # Should block low confidence queries
        self.assertEqual(confidence, 0.55)  # Confidence should be 0.55
    
    @patch('scripts.Hybrid_scripts.item_05_chain_retriever.model')
    def test_bayes_filter_exception_handling(self, mock_model):
        """Test filter exception handling"""
        # Configure the mock model to raise an exception
        mock_model.predict_proba.side_effect = Exception("Test exception")
        
        is_allowed, confidence = bayes_filter("test query")
        
        self.assertTrue(is_allowed)  # Should allow queries when exceptions occur
        self.assertEqual(confidence, 0.0)  # Confidence should be 0

if __name__ == '__main__':
    unittest.main() 