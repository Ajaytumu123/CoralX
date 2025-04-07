import os
import sys
import unittest
from pathlib import Path
import networkx as nx
import spacy
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from dotenv import load_dotenv, find_dotenv

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

class TestHybridDatabaseCreation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get the current script's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate one level up to the Coral-AI root directory
        cls.working_dir = os.path.abspath(os.path.join(current_dir, '..'))
        
        # Load environment variables
        load_dotenv(find_dotenv())
        
        # Define paths
        cls.pdf_folder = os.path.join(cls.working_dir, "data", "nine_pdfs")
        cls.faiss_save_path = os.path.join(cls.working_dir, "hybrid_index")
        cls.graph_path = os.path.join(cls.working_dir, "hybrid_index", "networkx_graph.pkl")
        
        # Initialize OpenAI embeddings
        cls.embedding = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Load spaCy
        try:
            cls.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            cls.nlp = spacy.load("en_core_web_sm")

    def test_1_required_directories_exist(self):
        """Test if required directories exist"""
        self.assertTrue(os.path.exists(self.pdf_folder), "PDF folder does not exist")
        self.assertTrue(os.path.exists(os.path.dirname(self.faiss_save_path)), 
                       "FAISS save directory does not exist")

    def test_2_pdf_files_exist(self):
        """Test if PDF files exist in the data directory"""
        pdf_files = list(Path(self.pdf_folder).glob("**/*.pdf"))
        self.assertGreater(len(pdf_files), 0, "No PDF files found in the data directory")

    def test_3_spacy_model_loaded(self):
        """Test if spaCy model is loaded correctly"""
        self.assertIsNotNone(self.nlp, "spaCy model is not loaded")
        test_text = "This is a test sentence."
        doc = self.nlp(test_text)
        self.assertGreater(len(list(doc.sents)), 0, "spaCy sentence parsing not working")

    def test_4_domain_entity_extraction(self):
        """Test domain-specific entity extraction"""
        test_text = "The Acropora coral species is affected by bleaching."
        doc = self.nlp(test_text)
        entities = []
        for ent in doc.ents:
            entities.append({"text": ent.text, "label": ent.label_})
        
        # Check for coral-specific terms
        coral_terms = ["acropora", "coral", "bleaching"]
        found_terms = [term for term in coral_terms if any(term in ent["text"].lower() for ent in entities)]
        self.assertGreater(len(found_terms), 0, "No coral-specific terms found")

    def test_5_semantic_document_splitting(self):
        """Test semantic document splitting"""
        loader = DirectoryLoader(self.pdf_folder, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        self.assertGreater(len(documents), 0, "No documents were loaded")
        
        # Test semantic splitting
        from scripts.Hybrid_scripts.item_01_database_creation_Hybrid import split_documents_semantically
        max_tokens = 500
        chunks = split_documents_semantically(documents, max_tokens=max_tokens)
        self.assertGreater(len(chunks), 0, "No chunks were created")
        
        # Verify chunk properties
        for chunk in chunks:
            self.assertIn('page_content', chunk.__dict__, "Chunk missing page_content")
            self.assertIn('metadata', chunk.__dict__, "Chunk missing metadata")
            # Only check if chunk size is within the max_tokens limit
            chunk_size = len(self.nlp(chunk.page_content))
            self.assertLessEqual(chunk_size, max_tokens, f"Chunk exceeds max tokens limit of {max_tokens}")

    def test_6_graph_creation_and_operations(self):
        """Test NetworkX graph creation and operations"""
        from scripts.Hybrid_scripts.item_01_database_creation_Hybrid import NetworkXGraph
        
        # Create a new graph
        graph = NetworkXGraph()
        self.assertIsInstance(graph.graph, nx.DiGraph, "Graph is not a directed graph")
        
        # Test document addition
        test_doc_id = "test_doc_1"
        test_content = "The Acropora coral is affected by ocean warming."
        graph.add_document(test_doc_id, test_content)
        
        # Verify graph structure
        self.assertIn(test_doc_id, graph.graph.nodes, "Document node not added to graph")
        self.assertGreater(len(graph.graph.edges), 0, "No edges created in graph")
        
        # Test graph saving and loading
        graph.save_graph(self.graph_path)
        self.assertTrue(os.path.exists(self.graph_path), "Graph file not created")
        
        loaded_graph = NetworkXGraph.load_graph(self.graph_path)
        self.assertIsInstance(loaded_graph.graph, nx.DiGraph, "Loaded graph is not a directed graph")
        self.assertIn(test_doc_id, loaded_graph.graph.nodes, "Document node not preserved in loaded graph")

    def test_7_hybrid_index_creation(self):
        """Test hybrid index creation with both FAISS and graph"""
        from scripts.Hybrid_scripts.item_01_database_creation_Hybrid import NetworkXGraph, split_documents_semantically
        
        # Load documents
        loader = DirectoryLoader(self.pdf_folder, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        # Create semantic chunks
        chunks = split_documents_semantically(documents)
        
        # Create FAISS index
        vectordb = FAISS.from_documents(documents=chunks, embedding=self.embedding)
        self.assertIsNotNone(vectordb, "FAISS index was not created")
        
        # Create and populate graph
        graph = NetworkXGraph()
        for i, chunk in enumerate(chunks):
            graph.add_document(f"doc_{i}", chunk.page_content)
        
        # Save both indices
        vectordb.save_local(self.faiss_save_path)
        graph.save_graph(self.graph_path)
        
        # Verify both indices were saved
        self.assertTrue(os.path.exists(self.faiss_save_path), "FAISS index was not saved")
        self.assertTrue(os.path.exists(self.graph_path), "Graph was not saved")

    def test_8_graph_querying(self):
        """Test graph querying functionality"""
        from scripts.Hybrid_scripts.item_01_database_creation_Hybrid import NetworkXGraph
        
        # Load the graph
        graph = NetworkXGraph.load_graph(self.graph_path)
        
        # Test query
        query = "coral bleaching effects"
        results = graph.query_graph(query)
        
        self.assertIsInstance(results, list, "Query results should be a list")
        if results:  # If any results were found
            self.assertIn("id", results[0], "Result missing document ID")
            self.assertIn("relevance_score", results[0], "Result missing relevance score")

if __name__ == '__main__':
    unittest.main(verbosity=2) 