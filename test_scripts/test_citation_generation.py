import os
import sys
import unittest
import pandas as pd
import subprocess
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

class TestCitationGeneration(unittest.TestCase):
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
        
        # Run the citation generation script first
        citation_script = os.path.join(cls.working_dir, "scripts", "Hybrid_scripts", "item_02_generate_citations_APA_FAISS.py")
        try:
            subprocess.run([sys.executable, citation_script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Citation generation script failed: {e}")
        except Exception as e:
            print(f"Warning: Error running citation generation script: {e}")

    def test_1_required_directories_exist(self):
        """Test if required directories exist"""
        self.assertTrue(os.path.exists(self.faiss_save_path), "FAISS index directory does not exist")
        self.assertTrue(os.path.exists(self.additional_files_dir), "Additional files directory does not exist")

    def test_2_faiss_index_loaded(self):
        """Test if FAISS index can be loaded"""
        try:
            vectordb = FAISS.load_local(self.faiss_save_path, self.embedding, allow_dangerous_deserialization=True)
            self.assertIsNotNone(vectordb, "FAISS index could not be loaded")
            self.assertGreater(len(vectordb.docstore.__dict__['_dict']), 0, "FAISS index is empty")
        except Exception as e:
            self.fail(f"Failed to load FAISS index: {str(e)}")

    def test_3_document_metadata(self):
        """Test if documents have required metadata"""
        vectordb = FAISS.load_local(self.faiss_save_path, self.embedding, allow_dangerous_deserialization=True)
        data_to_manipulate = vectordb.docstore.__dict__['_dict']
        
        for doc in data_to_manipulate.values():
            self.assertIn('metadata', doc.__dict__, "Document missing metadata")
            self.assertIn('source', doc.metadata, "Document metadata missing source")
            # Only check for citation if citations file exists
            if os.path.exists(self.citations_file):
                self.assertIn('citation', doc.metadata, "Document metadata missing citation")

    def test_4_citations_file_created(self):
        """Test if citations CSV file is created"""
        self.assertTrue(os.path.exists(self.citations_file), "Citations CSV file not created")

    def test_5_citations_file_format(self):
        """Test if citations CSV file has correct format"""
        if os.path.exists(self.citations_file):
            df = pd.read_csv(self.citations_file)
            self.assertIn('Source', df.columns, "CSV missing Source column")
            self.assertIn('Reference', df.columns, "CSV missing Reference column")
            self.assertGreater(len(df), 0, "CSV file is empty")

    def test_6_citation_format(self):
        """Test if citations follow APA format"""
        if os.path.exists(self.citations_file):
            df = pd.read_csv(self.citations_file)
            valid_citations = 0
            for _, row in df.iterrows():
                citation = row['Reference']
                # Skip citations that couldn't be generated
                if citation in ["I do not know.", "Citation generation failed", "No citation available"]:
                    continue
                
                # Basic APA format checks for valid citations
                self.assertTrue(
                    any(char.isdigit() for char in citation) or "(n.d.)" in citation,
                    f"Citation missing year or n.d.: {citation}"
                )
                self.assertTrue(any(char.isupper() for char in citation), 
                              f"Citation missing author name: {citation}")
                valid_citations += 1
            
            # Ensure we have at least some valid citations
            self.assertGreater(valid_citations, 0, 
                             "No valid citations found in the file")

    def test_7_citation_consistency(self):
        """Test if citations are consistent between FAISS index and CSV file"""
        if os.path.exists(self.citations_file):
            # Load citations from CSV
            df = pd.read_csv(self.citations_file)
            citations_dict = dict(zip(df['Source'], df['Reference']))
            
            # Load FAISS index
            vectordb = FAISS.load_local(self.faiss_save_path, self.embedding, allow_dangerous_deserialization=True)
            data_to_manipulate = vectordb.docstore.__dict__['_dict']
            
            # Check consistency
            for doc in data_to_manipulate.values():
                source = doc.metadata['source']
                if source in citations_dict:
                    self.assertIn('citation', doc.metadata, f"Citation missing for source: {source}")
                    self.assertEqual(doc.metadata['citation'], citations_dict[source], 
                                   f"Citation mismatch for source: {source}")

if __name__ == '__main__':
    unittest.main(verbosity=2) 