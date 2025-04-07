import os
import sys
import unittest
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from dotenv import load_dotenv, find_dotenv

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

# Import the module directly to test code flow
sys.path.append(os.path.join(project_root, 'scripts', 'Hybrid_scripts'))
import item_07_deep_research

class TestDeepResearchFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get the current script's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate one level up to the Coral-AI root directory
        cls.working_dir = os.path.abspath(os.path.join(current_dir, '..'))
        
        # Load environment variables
        load_dotenv(find_dotenv())
        
        # Define paths
        cls.faiss_save_path = os.path.join(cls.working_dir, "hybrid_index")
        cls.graph_path = os.path.join(cls.faiss_save_path, "networkx_graph.pkl")
        
        # Initialize OpenAI embeddings and LLM
        cls.embedding = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        cls.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def test_1_module_imports(self):
        """Test if all module imports work"""
        # Just importing the module is sufficient to test this
        self.assertTrue(hasattr(item_07_deep_research, 'NetworkXGraph'), "NetworkXGraph class not found")
        self.assertTrue(hasattr(item_07_deep_research, 'HybridRetriever'), "HybridRetriever class not found")
        self.assertTrue(hasattr(item_07_deep_research, 'ContextHybridRetriever'), "ContextHybridRetriever class not found")
        
    def test_2_networkx_graph_initialization(self):
        """Test NetworkX graph initialization"""
        try:
            graph = item_07_deep_research.NetworkXGraph()
            self.assertIsNotNone(graph, "Failed to initialize NetworkXGraph")
        except Exception as e:
            self.fail(f"NetworkXGraph initialization raised exception: {e}")

    def test_3_text_preprocessing(self):
        """Test text preprocessing function"""
        try:
            processed = item_07_deep_research.preprocess("Test")
            self.assertIsNotNone(processed, "Preprocess function returned None")
        except Exception as e:
            self.fail(f"Preprocess function raised exception: {e}")

    def test_4_networkx_graph_loading(self):
        """Test NetworkX graph loading if file exists"""
        if os.path.exists(self.graph_path):
            try:
                graph = item_07_deep_research.NetworkXGraph.load_graph(self.graph_path)
                self.assertIsNotNone(graph, "Failed to load NetworkXGraph")
            except Exception as e:
                self.fail(f"NetworkXGraph loading raised exception: {e}")
        else:
            self.skipTest("Graph file does not exist, skipping load test")

    def test_5_hybrid_retriever_initialization(self):
        """Test HybridRetriever initialization if FAISS index exists"""
        if os.path.exists(self.faiss_save_path):
            try:
                # Load FAISS index
                vectordb = FAISS.load_local(self.faiss_save_path, self.embedding, allow_dangerous_deserialization=True)
                
                # Initialize graph
                graph = item_07_deep_research.NetworkXGraph()
                
                # Initialize hybrid retriever
                retriever = item_07_deep_research.HybridRetriever(vectordb, graph)
                self.assertIsNotNone(retriever, "Failed to initialize HybridRetriever")
            except Exception as e:
                self.fail(f"HybridRetriever initialization raised exception: {e}")
        else:
            self.skipTest("FAISS index does not exist, skipping HybridRetriever initialization test")

    def test_6_context_hybrid_retriever_initialization(self):
        """Test ContextHybridRetriever initialization if FAISS index exists"""
        if os.path.exists(self.faiss_save_path):
            try:
                # Load FAISS index
                vectordb = FAISS.load_local(self.faiss_save_path, self.embedding, allow_dangerous_deserialization=True)
                
                # Initialize graph
                graph = item_07_deep_research.NetworkXGraph()
                
                # Initialize context hybrid retriever
                retriever = item_07_deep_research.ContextHybridRetriever(vectordb, graph, max_depth=1)
                self.assertIsNotNone(retriever, "Failed to initialize ContextHybridRetriever")
            except Exception as e:
                self.fail(f"ContextHybridRetriever initialization raised exception: {e}")
        else:
            self.skipTest("FAISS index does not exist, skipping ContextHybridRetriever initialization test")

if __name__ == '__main__':
    unittest.main(verbosity=2) 