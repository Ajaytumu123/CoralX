# Coral AI - Hybrid Scripts

This README provides documentation for the Coral AI hybrid retrieval system scripts and their associated test files. The hybrid system combines vector-based (FAISS) and graph-based (NetworkX) retrieval for more comprehensive coral research information retrieval.

## Overview of Hybrid Scripts

The hybrid scripts in this project implement a sophisticated information retrieval system combining semantic search with graph-based knowledge representation. These scripts form the backbone of the Coral AI knowledge system.

### Key Scripts

1. **item_01_database_creation_Hybrid.py**
   - Creates a hybrid database with both FAISS vector index and NetworkX graph components
   - Extracts domain-specific entities from coral research documents
   - Builds connections between documents and entities in the graph
   - Splits documents semantically to preserve context
   - Saves both the FAISS index and graph for later use

2. **item_02_generate_citation_APA_FAISS.py**
   - Generates APA citations for all documents stored in the FAISS index
   - Creates a CSV file mapping document sources to their corresponding APA citations
   - Enhances document retrieval with proper academic referencing

3. **item_03_replace_source_by_citation.py**
   - Updates document metadata in the FAISS index to use APA citations
   - Replaces raw source paths with properly formatted academic citations
   - Ensures all retrieved information can be properly attributed

4. **item_07_deep_research.py**
   - Implements an advanced research capability using the hybrid retrieval system
   - Features context-aware query expansion for more comprehensive results
   - Includes relevance checking to ensure queries are coral-related
   - Uses sub-query generation to explore related topics
   - Provides document deduplication to ensure diverse results

## Running the Hybrid Scripts

To use these scripts, follow the instructions below. Make sure you have all required dependencies installed.

### Prerequisites

1. Python 3.11.5 environment
2. Required packages installed (see `requirements.txt`)
3. OpenAI API key set in your environment variables or .env file
4. PDF documents in the data directory

### Setup

1. Clone the repository and navigate to the project root:
   ```bash
   git clone [repository-url]
   cd Coral-AI
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Running the Scripts

**1. Create the Hybrid Database:**
```bash
python scripts/Hybrid_scripts/item_01_database_creation_Hybrid.py
```
This will:
- Process PDF files in the data directory
- Create a FAISS vector index
- Build a NetworkX graph connecting documents and entities
- Save both to the hybrid_index directory

**2. Generate APA Citations:**
```bash
python scripts/Hybrid_scripts/item_02_generate_citation_APA_FAISS.py
```
This will:
- Process documents in the FAISS index
- Generate APA format citations for each document
- Save a mapping CSV file in additional_files/citations.csv

**3. Replace Sources with Citations:**
```bash
python scripts/Hybrid_scripts/item_03_replace_source_by_citation.py
```
This will:
- Update the FAISS index with proper citations
- Replace raw file paths with academic citations

**4. Run Deep Research:**
```bash
python scripts/Hybrid_scripts/item_07_deep_research.py
```
This will:
- Launch the deep research capability
- Allow you to enter research queries
- Return comprehensive results using the hybrid system
- Generate follow-up questions automatically

### Script Execution Order

For proper functionality, execute the scripts in this order:
1. item_01_database_creation_Hybrid.py
2. item_02_generate_citation_APA_FAISS.py
3. item_03_replace_source_by_citation.py
4. item_07_deep_research.py

## Test Scripts

The test scripts validate the functionality of the hybrid system components. These tests ensure that the code is working as expected and that the data structures are properly maintained.

### Available Test Scripts

1. **test_hybrid_database_creation.py**
   - Tests the database creation functionality
   - Validates directory structure, document loading, entity extraction, and graph creation
   - Ensures semantic document splitting works correctly

2. **test_citation_generation.py**
   - Tests the citation generation functionality
   - Validates FAISS index loading, document metadata processing, and citation formatting
   - Checks that citations follow APA format

3. **test_source_replacement.py**
   - Tests the source replacement functionality
   - Ensures citations are properly integrated into document metadata
   - Checks that document content integrity is maintained

4. **test_deep_research.py**
   - Tests the deep research code flow
   - Validates NetworkX graph operations, retriever initialization, and content processing
   - Checks that the hybrid retrieval system can be properly initialized

### Running the Tests

To run the tests, use the Python unittest module:

**Run all tests:**
```bash
python -m unittest discover test_scripts
```

**Run a specific test:**
```bash
python -m unittest test_scripts/test_hybrid_database_creation.py
```

**Run tests with verbose output:**
```bash
python -m unittest test_scripts/test_deep_research.py -v
```

## Troubleshooting

### Common Issues

1. **Module not found errors:**
   Make sure you're running the scripts from the project root directory. The scripts use relative imports.

2. **FAISS index not found:**
   Ensure you've run the database creation script first (item_01_database_creation_Hybrid.py).

3. **API key errors:**
   Check that your .env file is properly set up with your OpenAI API key.

4. **Memory issues:**
   Processing large PDF collections may require substantial memory. Consider reducing batch sizes or upgrading your hardware.

### Note on DeprecationWarnings

You may see DeprecationWarnings related to SwigPy types when running the FAISS operations. These are internal warnings from the FAISS library and can be safely ignored as they don't affect functionality.

## Further Resources

For more information about the underlying technologies:

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [NetworkX Documentation](https://networkx.org/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [OpenAI API Documentation](https://platform.openai.com/docs/introduction) 