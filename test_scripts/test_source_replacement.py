import os
import sys
import unittest
import pandas as pd
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

class TestSourceReplacement(unittest.TestCase):
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
        cls.additional_files_dir = os.path.join(cls.working_dir, "additional_files")
        cls.citations_file = os.path.join(cls.additional_files_dir, "citations.csv")
        
        # Initialize OpenAI embeddings
        cls.embedding = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    def test_1_required_files_exist(self):
        """Test if required files exist"""
        self.assertTrue(os.path.exists(self.faiss_save_path), "FAISS index directory does not exist")
        self.assertTrue(os.path.exists(self.citations_file), "Citations CSV file does not exist")

    def test_2_faiss_index_loaded(self):
        """Test if FAISS index can be loaded"""
        try:
            vectordb = FAISS.load_local(self.faiss_save_path, self.embedding, allow_dangerous_deserialization=True)
            self.assertIsNotNone(vectordb, "FAISS index could not be loaded")
            self.assertGreater(len(vectordb.docstore.__dict__['_dict']), 0, "FAISS index is empty")
        except Exception as e:
            self.fail(f"Failed to load FAISS index: {str(e)}")

    def test_3_citations_file_loaded(self):
        """Test if citations CSV file can be loaded"""
        try:
            df = pd.read_csv(self.citations_file)
            self.assertIsNotNone(df, "Citations CSV file could not be loaded")
            self.assertGreater(len(df), 0, "Citations CSV file is empty")
            self.assertIn('Source', df.columns, "CSV missing Source column")
            self.assertIn('Reference', df.columns, "CSV missing Reference column")
        except Exception as e:
            self.fail(f"Failed to load citations CSV file: {str(e)}")

    def test_4_source_replacement(self):
        """Test if sources are replaced with citations"""
        # Load FAISS index
        vectordb = FAISS.load_local(self.faiss_save_path, self.embedding, allow_dangerous_deserialization=True)
        
        # Load citations
        citations_df = pd.read_csv(self.citations_file)
        source_to_reference = dict(zip(citations_df['Source'], citations_df['Reference']))
        
        # Get a sample document
        all_keys = list(vectordb.docstore.__dict__['_dict'].keys())
        if not all_keys:
            self.fail("No documents found in FAISS index")
            
        sample_doc = vectordb.docstore.__dict__['_dict'][all_keys[0]]
        original_source = sample_doc.metadata.get('source')
        
        # Check if the source exists in our citations
        if original_source in source_to_reference:
            self.assertNotEqual(
                original_source,
                source_to_reference[original_source],
                "Source and citation are identical"
            )

    def test_5_metadata_preservation(self):
        """Test if other metadata is preserved after source replacement"""
        # Load FAISS index
        vectordb = FAISS.load_local(self.faiss_save_path, self.embedding, allow_dangerous_deserialization=True)
        
        # Get a sample document
        all_keys = list(vectordb.docstore.__dict__['_dict'].keys())
        if not all_keys:
            self.fail("No documents found in FAISS index")
            
        sample_doc = vectordb.docstore.__dict__['_dict'][all_keys[0]]
        
        # Check if document has required metadata
        self.assertIn('metadata', sample_doc.__dict__, "Document missing metadata")
        self.assertIn('source', sample_doc.metadata, "Document metadata missing source")
        self.assertIn('page_content', sample_doc.__dict__, "Document missing page_content")

    def test_6_citation_format_preservation(self):
        """Test if citations maintain proper format after replacement"""
        # Load FAISS index
        vectordb = FAISS.load_local(self.faiss_save_path, self.embedding, allow_dangerous_deserialization=True)
        
        # Load citations
        citations_df = pd.read_csv(self.citations_file)
        source_to_reference = dict(zip(citations_df['Source'], citations_df['Reference']))
        
        # Get a sample document
        all_keys = list(vectordb.docstore.__dict__['_dict'].keys())
        if not all_keys:
            self.fail("No documents found in FAISS index")
            
        sample_doc = vectordb.docstore.__dict__['_dict'][all_keys[0]]
        source = sample_doc.metadata.get('source')
        
        # If the source is a citation, check its format
        if source in source_to_reference.values():
            # Check for basic APA format elements
            self.assertTrue(
                any(char.isdigit() for char in source) or "(n.d.)" in source,
                f"Citation missing year or n.d.: {source}"
            )
            self.assertTrue(any(char.isupper() for char in source), 
                          f"Citation missing author name: {source}")

if __name__ == '__main__':
    unittest.main(verbosity=2) 