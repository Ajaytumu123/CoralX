import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the functions to test
from scripts.Hybrid_scripts.item_05_chain_retriever import answer_to_QA, bayes_filter, log_unrelated_query

class TestQueryProcessing(unittest.TestCase):
    """Integration tests for the query processing flow"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create patch for model and other dependencies
        self.model_patcher = patch('scripts.Hybrid_scripts.item_05_chain_retriever.model')
        self.mock_model = self.model_patcher.start()
        
        # Create patch for FAISS vectorstore
        self.faiss_patcher = patch('langchain_community.vectorstores.FAISS.load_local')
        self.mock_faiss = self.faiss_patcher.start()
        
        # Create patch for OpenAI
        self.openai_patcher = patch('langchain_openai.ChatOpenAI')
        self.mock_openai = self.openai_patcher.start()
        
        # Create patch for input function to simulate user responses
        self.input_patcher = patch('builtins.input')
        self.mock_input = self.input_patcher.start()
        
        # Configure mocks
        self.mock_model.predict_proba.return_value = [[0.3, 0.7]]  # Default to relevant
        self.mock_model.classes_ = ["irrelevant", "relevant"]
        
        # Setup mock retriever, vectordb and chain
        self.mock_vectordb = MagicMock()
        self.mock_retriever = MagicMock()
        self.mock_vectordb.as_retriever.return_value = self.mock_retriever
        self.mock_faiss.return_value = self.mock_vectordb
        
        # Setup mock LLM response
        self.mock_llm = MagicMock()
        self.mock_openai.return_value = self.mock_llm
        
        # Mock RetrievalQA
        self.qa_chain_patcher = patch('langchain.chains.RetrievalQA.from_chain_type')
        self.mock_qa_chain = self.qa_chain_patcher.start()
        
        # Setup response
        self.mock_qa_result = {
            'result': 'Test answer about coral reefs.\n\nAdditional information here.',
            'source_documents': [
                MagicMock(metadata={'source': 'test_source_1.pdf'}),
                MagicMock(metadata={'source': 'test_source_2.pdf'})
            ]
        }
        self.mock_qa_chain.return_value.invoke.return_value = self.mock_qa_result
    
    def tearDown(self):
        """Tear down test fixtures"""
        self.model_patcher.stop()
        self.faiss_patcher.stop()
        self.openai_patcher.stop()
        self.input_patcher.stop()
        self.qa_chain_patcher.stop()
    
    @patch('scripts.Hybrid_scripts.item_05_chain_retriever.log_unrelated_query')
    def test_relevant_query_processing(self, mock_log):
        """Test processing a relevant query"""
        # Configure model to predict as relevant
        self.mock_model.predict_proba.return_value = [[0.2, 0.8]]  # 80% relevant
        
        # Process a relevant query
        result = answer_to_QA("How do coral reefs form?")
        
        # Verify behavior
        self.mock_model.predict_proba.assert_called_once()
        mock_log.assert_not_called()  # Should not log relevant queries
        self.mock_qa_chain.return_value.invoke.assert_called_once()
        self.assertIn("Test answer about coral reefs", result)
        self.assertIn("test_source_1.pdf", result)
        self.assertIn("test_source_2.pdf", result)
    
    @patch('scripts.Hybrid_scripts.item_05_chain_retriever.log_unrelated_query')
    def test_irrelevant_query_rejected(self, mock_log):
        """Test rejecting an irrelevant query"""
        # Configure model to predict as irrelevant
        self.mock_model.predict_proba.return_value = [[0.9, 0.1]]  # 90% irrelevant
        
        # Simulate user rejecting the query
        self.mock_input.return_value = "no"
        
        # Process an irrelevant query
        result = answer_to_QA("Best chocolate cake recipe")
        
        # Verify behavior
        self.mock_model.predict_proba.assert_called_once()
        mock_log.assert_called_once_with("Best chocolate cake recipe", 0.9, False)
        self.mock_qa_chain.return_value.invoke.assert_not_called()  # Should not process
        self.assertIn("QUERY CANCELLED", result)
        self.assertIn("CoralX Foundation", result)
    
    @patch('scripts.Hybrid_scripts.item_05_chain_retriever.log_unrelated_query')
    def test_irrelevant_query_approved(self, mock_log):
        """Test approving an irrelevant query"""
        # Configure model to predict as irrelevant
        self.mock_model.predict_proba.return_value = [[0.9, 0.1]]  # 90% irrelevant
        
        # Simulate user approving the query
        self.mock_input.return_value = "yes"
        
        # Process an irrelevant query
        result = answer_to_QA("How to make pasta")
        
        # Verify behavior
        self.mock_model.predict_proba.assert_called_once()
        mock_log.assert_called_once_with("How to make pasta", 0.9, True)
        self.mock_qa_chain.return_value.invoke.assert_called_once()
        self.assertIn("Test answer about coral reefs", result)

if __name__ == '__main__':
    unittest.main() 