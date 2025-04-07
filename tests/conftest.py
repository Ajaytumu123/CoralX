"""
Pytest configuration for the CoralAI test suite.

This module contains fixtures and configuration settings for running the test suite.
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import fixtures and mocks
from tests.fixtures.sample_queries import CORAL_QUERIES, IRRELEVANT_QUERIES, AMBIGUOUS_QUERIES
from tests.fixtures.mock_responses import (
    MOCK_DOCUMENTS, 
    MOCK_CORAL_RESPONSE, 
    MOCK_GENERAL_RESPONSE,
    get_mock_retriever,
    get_mock_vectorstore
)

@pytest.fixture
def mock_model():
    """Fixture providing a mock Bayesian filter model"""
    mock = MagicMock()
    mock.classes_ = ["irrelevant", "relevant"]
    
    # Default to classifying as relevant
    mock.predict_proba.return_value = [[0.2, 0.8]]  # 80% relevant, 20% irrelevant
    
    return mock

@pytest.fixture
def mock_openai():
    """Fixture providing a mock OpenAI LLM"""
    mock = MagicMock()
    
    # Configure the mock to return a standard response
    mock.invoke.return_value.content = "Mock LLM response about coral reefs."
    
    return mock

@pytest.fixture
def mock_vectordb():
    """Fixture providing a mock vector database"""
    return get_mock_vectorstore()

@pytest.fixture
def sample_documents():
    """Fixture providing sample document objects"""
    return MOCK_DOCUMENTS

@pytest.fixture
def mock_qa_chain():
    """Fixture providing a mock QA chain"""
    mock = MagicMock()
    
    # Configure the mock to return a standard response
    mock.invoke.return_value = MOCK_CORAL_RESPONSE
    
    return mock

@pytest.fixture
def test_env_setup():
    """Fixture for setting up the test environment with all necessary mocks"""
    # Create patches for all external dependencies
    patches = [
        patch('scripts.Hybrid_scripts.item_05_chain_retriever.model', MagicMock()),
        patch('langchain_community.vectorstores.FAISS.load_local', return_value=get_mock_vectorstore()),
        patch('langchain_openai.ChatOpenAI', return_value=MagicMock()),
        patch('langchain.chains.RetrievalQA.from_chain_type', return_value=MagicMock()),
        patch('builtins.input', return_value="yes")  # Default to approving queries
    ]
    
    # Start all patches
    mocks = [p.start() for p in patches]
    
    # Configure the model mock to classify as relevant by default
    mocks[0].predict_proba.return_value = [[0.2, 0.8]]
    mocks[0].classes_ = ["irrelevant", "relevant"]
    
    # Configure the QA chain to return a standard response
    mocks[3].return_value.invoke.return_value = MOCK_CORAL_RESPONSE
    
    # Yield to let the test run
    yield {
        'model': mocks[0],
        'vectorstore': mocks[1],
        'llm': mocks[2],
        'qa_chain': mocks[3],
        'input': mocks[4]
    }
    
    # Stop all patches
    for p in patches:
        p.stop()

# Register markers
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "system: marks tests as system tests")
    config.addinivalue_line("markers", "model: tests requiring the actual model file")
    config.addinivalue_line("markers", "openai: tests requiring OpenAI API access") 