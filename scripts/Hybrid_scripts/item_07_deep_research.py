import os
import re
import sys
import urllib.parse
import time
import warnings
import joblib
import datetime
import networkx as nx
from dotenv import load_dotenv, find_dotenv
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, SkipValidation

# Suppress all warnings to maintain a clean output
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

# LangChain imports for vector storage and retrieval
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.globals import set_verbose, set_debug
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate

# Set working directory for file operations
current_dir = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Load environment variables from a .env file
load_dotenv(find_dotenv())
set_verbose(False)
set_debug(False)

# Constants for model and file paths
LLM_MODEL = "gpt-4o-mini"  # Model identifier for the language model
VECTOR_STORE_PATH = "hybrid_index"  # Path to the vector store
GRAPH_PATH = "hybrid_index/networkx_graph.pkl"  # Path to the graph data

# Global variable to hold the content filter model
model = None

class NetworkXGraph:
    """Local implementation of graph-based retrieval using NetworkX.

    This class encapsulates a graph structure and provides methods to load
    a graph from a file and query it for relevant documents based on node
    degrees.

    Attributes:
        graph (nx.Graph): The underlying graph structure for document retrieval.
    """
    
    def __init__(self):
        """Initialize a new empty graph."""
        self.graph = nx.Graph()
        
    @classmethod
    def load_graph(cls, path: str):
        """Load a graph from a pickle file.

        Args:
            path (str): The file path to the pickle file containing the graph.

        Returns:
            NetworkXGraph: An instance of NetworkXGraph containing the loaded graph,
            or a new instance with an empty graph if loading fails.
        """
        try:
            graph = cls()  # Create a new instance of NetworkXGraph
            graph.graph = nx.read_gpickle(path)  # Load the graph from the specified path
            return graph
        except Exception as e:
            return cls()  # Return an empty graph on failure
    
    def query_graph(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Query the graph for relevant documents based on the input query.

        This method retrieves the top k nodes with the highest degree, which
        indicates their relevance.

        Args:
            query (str): The query string to search for in the graph.
            k (int): The number of top nodes to return (default is 4).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the node IDs
            and their relevance scores.
        """
        try:
            # Sort nodes by degree and select the top k
            nodes = sorted(self.graph.nodes(data=True), 
                         key=lambda x: self.graph.degree(x[0]), 
                         reverse=True)[:k]
            
            # Return a list of dictionaries with node IDs and their relevance scores
            return [{"id": node[0], 
                     "relevance_score": self.graph.degree(node[0]) / max(1, max(dict(self.graph.degree()).values()))}
                    for node in nodes]
        except Exception as e:
            return []  # Return an empty list on failure

def log_unrelated_query(query, confidence, proceeded):
    """Log unrelated queries for review.

    This function records queries that are deemed unrelated to coral research,
    along with their confidence scores and whether the user chose to proceed.

    Args:
        query (str): The user query that was logged.
        confidence (float): The confidence score from the filter indicating
        the relevance of the query.
        proceeded (bool): Indicates whether the user chose to proceed with the query.
    """
    try:
        log_dir = os.path.join(working_dir, 'logs')  # Directory for log files
        os.makedirs(log_dir, exist_ok=True)  # Create log directory if it doesn't exist
        
        log_file = os.path.join(log_dir, 'unrelated_queries.log')  # Log file path
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Current timestamp
        
        # Append log entry to the log file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] Query: {query}\n")
            f.write(f"Confidence: {confidence:.2f}\n")
            f.write(f"User proceeded: {proceeded}\n")
            f.write("-" * 50 + "\n")
    except Exception as e:
        pass  # Ignore errors during logging

def preprocess(text):
    """Preprocess text for the Bayes filter.

    This function normalizes the input text by converting it to lowercase
    and stripping whitespace.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    return text.lower().strip()

def generate_subqueries(query: str, context_docs: List[Document], llm: ChatOpenAI) -> tuple[List[str], bool]:
    """Generate sub-queries considering existing context and check coral relevance.

    This function checks if the input query is related to coral research and
    generates follow-up sub-queries based on the context documents provided.

    Args:
        query (str): The original user query.
        context_docs (List[Document]): A list of context documents to inform sub-query generation.
        llm (ChatOpenAI): The language model instance used for generating sub-queries.

    Returns:
        tuple: A tuple containing:
            - List[str]: A list of generated queries (including the original).
            - bool: Indicates if the query is coral-related.
    """
    # First check if the query is coral-related
    relevance_prompt = f"""Determine if this query is specifically about coral research, coral preservation, 
    or directly related marine biology topics that affect coral ecosystems. Answer ONLY with 'yes' or 'no'.
    
    Query: {query}
    
    Is this query specifically about coral research, preservation, or directly related marine biology?: """
    
    relevance_check = llm.invoke(relevance_prompt).content.strip().lower()  # Invoke LLM for relevance check
    is_coral_relevant = relevance_check == 'yes'  # Determine if the query is relevant
    
    if not is_coral_relevant:
        print(f"\n{'='*40}\nQUERY RELEVANCE CHECK: Not coral-related\n{'='*40}")
        return [], False  # Return empty list if not relevant
        
    context_str = ""
    if context_docs:
        context_str = "Existing Context Information:\n"
        for i, doc in enumerate(context_docs[:3]):  # Limit to first 3 context documents
            content_preview = doc.page_content[:250].replace('\n', ' ')  # Preview content
            context_str += f"- Document {i+1}: {content_preview}...\n"
    
    # Generate sub-queries based on the context and original question
    prompt = f"""Analyze the following context and original question to generate exactly 2 follow-up 
    sub-questions that target missing information. Maintain the original question as the first line.
    Only generate sub-questions if they are specifically about coral research or preservation.

    {context_str}
    Original Question: {query}

    Generate 1 original question line followed by exactly 2 sub-questions on new lines:"""
    
    response = llm.invoke(prompt)  # Invoke LLM to generate sub-queries
    queries = []
    seen = set()  # Track seen queries to avoid duplicates
    
    for line in response.content.split('\n'):
        line = line.strip().strip('-*').strip()  # Clean up the line
        if line:
            lower_line = line.lower()
            if lower_line not in seen:  # Avoid duplicates
                seen.add(lower_line)
                queries.append(line)
    
    if query.lower() not in seen:  # Ensure the original query is included
        queries.insert(0, query)
    
    print(f"\n{'='*40}\nTHINKING: Generated queries with context awareness\n{'='*40}")
    for i, q in enumerate(queries):
        prefix = "Original" if i == 0 else f"Sub-query {i}"
        print(f"{prefix}: {q}")
    
    return queries[:3], True  # Return the first 3 queries and relevance status

class HybridRetriever(BaseRetriever):
    """Basic hybrid retriever for a single query.

    This class combines vector-based and graph-based retrieval methods to
    find relevant documents for a given query.

    Attributes:
        _vector_retriever (BaseRetriever): The vector store retriever instance.
        _graph (NetworkXGraph): The graph instance for document retrieval.
        _k (int): The number of top documents to retrieve.
    """
    
    def __init__(self, vector_store, graph, k=4):
        """Initialize the HybridRetriever with a vector store and graph.

        Args:
            vector_store: The vector store instance for document retrieval.
            graph (NetworkXGraph): The graph instance for document retrieval.
            k (int): The number of top documents to retrieve (default is 4).
        """
        super().__init__()  # Initialize the base class
        # Store these as instance variables not using pydantic fields
        self._vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})  # Vector retriever
        self._graph = graph  # Graph for hybrid retrieval
        self._k = k  # Number of documents to retrieve

    def _get_relevant_documents(self, query):
        """Retrieve relevant documents for the given query using hybrid search.

        This method combines results from both vector and graph searches,
        boosting documents found in both.

        Args:
            query (str): The query string to search for.

        Returns:
            List[Document]: A list of relevant documents sorted by relevance.
        """
        try:
            # Get vector search results
            vector_docs = self._vector_retriever.invoke(query)
            
            # Get graph search results
            graph_results = self._graph.query_graph(query)
            
            # Extract document IDs from graph results
            graph_doc_ids = {r["id"] for r in graph_results}
            
            # Boost documents found in both searches
            hybrid_docs = []
            for doc in vector_docs:
                if doc.metadata.get("doc_id") in graph_doc_ids:
                    doc.metadata["graph_enriched"] = True  # Mark as enriched by graph
                    # Add a relevance boost to documents found in both searches
                    doc.metadata["relevance_boost"] = graph_results[list(graph_doc_ids).index(doc.metadata["doc_id"])].get("relevance_score", 1)
                hybrid_docs.append(doc)
            
            # Sort by relevance if graph-enriched
            hybrid_docs.sort(key=lambda x: x.metadata.get("relevance_boost", 0), reverse=True)
            
            return hybrid_docs  # Return sorted documents
            
        except Exception as e:
            # Fallback to vector search if graph search fails
            return vector_docs if 'vector_docs' in locals() else []  # Return vector docs if available
            
    async def _aget_relevant_documents(self, query):
        """Asynchronously retrieve relevant documents for the given query.

        This method is a coroutine that allows for non-blocking retrieval.

        Args:
            query (str): The query string to search for.

        Returns:
            List[Document]: A list of relevant documents.
        """
        return self._get_relevant_documents(query)  # Call the synchronous method

class ContextHybridRetriever(BaseRetriever, BaseModel):
    """Iterative context-aware retriever with hybrid search.

    This class extends the hybrid retriever to support iterative querying
    based on context and depth of search.

    Attributes:
        base_hybrid_retriever (SkipValidation[Any]): The base hybrid retriever instance.
        max_depth (int): The maximum depth for iterative querying.
    """
    base_hybrid_retriever: SkipValidation[Any] = None
    max_depth: int = Field(default=2)  # Default maximum depth for queries
    
    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types in Pydantic model
        
    def __init__(self, vector_store, graph, k=4, max_depth=2):
        """Initialize the ContextHybridRetriever with a vector store and graph.

        Args:
            vector_store: The vector store instance for document retrieval.
            graph (NetworkXGraph): The graph instance for document retrieval.
            k (int): The number of top documents to retrieve (default is 4).
            max_depth (int): The maximum depth for iterative querying (default is 2).
        """
        # Initialize base retriever
        hybrid_retriever = HybridRetriever(vector_store, graph, k)
        
        # Initialize BaseRetriever
        BaseRetriever.__init__(self)
        
        # Set up attributes
        self.base_hybrid_retriever = hybrid_retriever  # Store the base hybrid retriever
        self.max_depth = max_depth  # Set maximum depth
        
        # Initialize BaseModel with our attributes
        BaseModel.__init__(
            self, 
            base_hybrid_retriever=self.base_hybrid_retriever, 
            max_depth=self.max_depth
        )
    
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """Retrieve relevant documents for the given query with iterative context awareness.

        This method processes the initial query and generates sub-queries
        iteratively based on the context documents retrieved.

        Args:
            query (str): The query string to search for.
            run_manager (Optional): An optional run manager for tracking progress.

        Returns:
            List[Document]: A list of unique relevant documents retrieved.
        """
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)  # Initialize the language model
        accumulated_docs = []  # List to accumulate retrieved documents
        all_queries = [query]  # List to track all queries
        
        print(f"\n{'='*40}\nRETRIEVAL PROCESS START\n{'='*40}")
        
        # Process initial query
        print(f"\nSEARCHING: Initial query: '{query}'")
        initial_docs = self.base_hybrid_retriever._get_relevant_documents(query)  # Get initial documents
        accumulated_docs.extend(initial_docs)  # Add to accumulated documents
        unique_docs = self._deduplicate(accumulated_docs)  # Deduplicate documents
        self._log_documents(initial_docs, "Initial query")  # Log initial documents
        
        current_queries = [query]  # Start with the initial query
        
        for depth in range(self.max_depth):
            new_subqueries = []  # List to hold new sub-queries
            print(f"\n{'='*40}\nEXPANSION DEPTH: {depth+1}\n{'='*40}")
            
            for q in current_queries:
                sub_queries, is_coral_relevant = generate_subqueries(q, unique_docs, llm)  # Generate sub-queries
                
                for sq in sub_queries:
                    if sq not in all_queries:  # Avoid duplicates
                        all_queries.append(sq)  # Add to all queries
                        print(f"\nSEARCHING: Follow-up query: '{sq}'")
                        docs = self.base_hybrid_retriever._get_relevant_documents(sq)  # Get documents for sub-query
                        self._log_documents(docs, sq)  # Log documents for sub-query
                        accumulated_docs.extend(docs)  # Add to accumulated documents
                        new_subqueries.append(sq)  # Track new sub-queries
            
            unique_docs = self._deduplicate(accumulated_docs)  # Deduplicate accumulated documents
            current_queries = new_subqueries  # Update current queries
            
            if not current_queries:  # Exit if no new queries
                break
        
        print(f"\n{'='*40}\nTOTAL UNIQUE DOCUMENTS: {len(unique_docs)}\n{'='*40}")
        return unique_docs  # Return unique documents
    
    def _deduplicate(self, docs: List[Document]) -> List[Document]:
        """Remove duplicate documents from the list.

        This method uses a hash-based approach to ensure that each document
        is unique based on its content and source.

        Args:
            docs (List[Document]): The list of documents to deduplicate.

        Returns:
            List[Document]: A list of unique documents.
        """
        seen = set()  # Set to track seen document hashes
        unique = []  # List to hold unique documents
        for doc in docs:
            # Create a unique hash for each document based on its source and content
            doc_hash = f"{doc.metadata.get('source', '')}-{hash(doc.page_content[:200])}"
            if doc_hash not in seen:  # Check if the document is unique
                seen.add(doc_hash)  # Mark as seen
                unique.append(doc)  # Add to unique list
        return unique  # Return unique documents
    
    def _log_documents(self, docs: List[Document], query: str):
        """Log the documents found for a specific query.

        This method prints the number of documents found and their previews
        to the console for review.

        Args:
            docs (List[Document]): The list of documents found.
            query (str): The query string for which documents were found.
        """
        print(f"FOUND {len(docs)} DOCUMENTS for '{query}':")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')  # Get document source
            content = doc.page_content[:150].replace('\n', ' ') + '...'  # Preview content
            print(f"[Doc {i} from {source}] {content}")  # Print document info
    
    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """Asynchronously retrieve relevant documents for the given query.

        This method is a coroutine that allows for non-blocking retrieval.

        Args:
            query (str): The query string to search for.
            run_manager (Optional): An optional run manager for tracking progress.

        Returns:
            List[Document]: A list of relevant documents.
        """
        return self._get_relevant_documents(query, run_manager=run_manager)  # Call the synchronous method

def process_llm_response_with_sources(llm_response):
    """Process the response from the language model ensuring answers are based only on provided context.

    This function splits the LLM response into a main answer and additional details,
    and formats the output to include sources.

    Args:
        llm_response (dict): The response from the language model containing the answer
        and source documents.

    Returns:
        str: The formatted final answer with sources.
    """
    # Split answer into main and additional details
    full_answer = llm_response['result'].strip()  # Clean up the answer
    parts = full_answer.split('\n\n', 1)  # Split into main answer and additional details
    
    # If no source documents, return a default message
    if not llm_response.get('source_documents'):
        return "FINAL ANSWER:\nI don't have enough reliable information in my knowledge base to answer this question."
    
    if len(parts) == 1:
        main_answer = parts[0]  # Only main answer
        additional_details = ""
    else:
        main_answer, additional_details = parts[0], parts[1]  # Separate parts
    
    # Handle uncertain answers
    main_lower = main_answer.lower()
    if any(phrase in main_lower for phrase in ["i don't know", "not sure", "no information"]):
        return "FINAL ANSWER:\n" + main_answer.capitalize()  # Return uncertain answer
    
    # Build final output
    final_output = "\n" + "="*40 + "\nFINAL ANSWER:\n" + "="*40 + "\n"
    final_output += f"{main_answer}\n\n"  # Add main answer
    if additional_details:
        final_output += f"Additional Details:\n{additional_details}\n\n"  # Add additional details if present
    
    # Add sources to the output
    unique_sources = set()  # Set to track unique sources
    for source in llm_response["source_documents"]:
        citation = source.metadata.get('source', 'Unknown source')  # Get source citation
        unique_sources.add(citation)  # Add to unique sources
    
    if unique_sources:
        final_output += "Sources:\n" + "\n".join(f"- {src}" for src in sorted(unique_sources))  # Format sources
    
    return final_output  # Return the final formatted output

def initialize_content_filter():
    """Initialize the content filter model.

    This function attempts to load a pre-trained Naive Bayes classifier model
    for filtering queries related to coral research.

    Returns:
        model: The loaded classifier model, or None if loading fails.
    """
    try:
        model_paths = [
            os.path.join(working_dir, 'BayesFilter', 'coral_classifier.joblib'),
            os.path.join(working_dir, 'coral_classifier.joblib')
        ]
        
        # Suppress all sklearn warnings during model loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            
            for path in model_paths:
                if os.path.exists(path):  # Check if the model file exists
                    model = joblib.load(path)  # Load the model
                    return model  # Return the loaded model
                    
        return None  # Return None if no model was loaded
    except Exception as e:
        print(f"Error loading model: {e}")  # Print error message
        return None  # Return None on error

def bayes_filter(query, irrelevant_threshold=0.7, min_confidence=0.6):
    """Filter queries using a Naive Bayes classifier to detect if they're related to coral research.

    This function evaluates the input query and determines if it should be allowed
    based on the classifier's predictions.

    Args:
        query (str): The user query to evaluate.
        irrelevant_threshold (float): Confidence threshold for marking as irrelevant (default is 0.7).
        min_confidence (float): Minimum confidence required for a reliable prediction (default is 0.6).

    Returns:
        tuple: A tuple containing:
            - bool: Indicates if the query is allowed (True) or blocked (False).
            - float: The confidence score of the prediction.
    """
    if model is None:
        return True, 0.0  # Allow all queries if model isn't available
        
    try:
        # Preprocess the query (using the function defined above)
        processed_query = preprocess(query)
        
        # Get prediction probabilities
        proba = model.predict_proba([processed_query])[0]  # Predict probabilities for the query
        confidence = proba.max()  # Get the maximum confidence score
        predicted_class = model.classes_[proba.argmax()]  # Get the predicted class
        
        print(f"Filter prediction: '{predicted_class}' with confidence {confidence:.2f}")
        
        # Block if either:
        # 1. Predicted irrelevant with high confidence
        # 2. Overall confidence is too low (uncertain prediction)
        if (predicted_class == "irrelevant" and confidence > irrelevant_threshold) \
            or (confidence < min_confidence):
            return False, confidence  # Block the query
        return True, confidence  # Allow the query
    except Exception as e:
        return True, 0.0  # Fail-safe: allow query through

def answer_to_QA(query: str, faiss_index_path: str = VECTOR_STORE_PATH):
    """Process a query using the RAG system with both LLM relevance check and Bayes filtering.

    This function orchestrates the retrieval of relevant documents and the generation
    of answers based on the input query, utilizing both a language model and a content
    filter.

    Args:
        query (str): The user query to process.
        faiss_index_path (str): Path to the FAISS index (default is VECTOR_STORE_PATH).

    Returns:
        str: The processed answer with sources.
    """
    print(f"\n{'='*40}\nPROCESSING QUESTION: {query}\n{'='*40}")
    
    # Initialize global model if needed
    global model
    if not globals().get('model'):
        model = initialize_content_filter()  # Load the content filter model
    
    # Initialize OpenAI and embeddings
    embedding = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))  # Create embeddings instance
    vectordb = FAISS.load_local(faiss_index_path, embedding, allow_dangerous_deserialization=True)  # Load FAISS index
    graph = NetworkXGraph.load_graph(GRAPH_PATH)  # Load the graph
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)  # Initialize the language model
    
    # First check with LLM if query is coral-related
    queries, is_coral_relevant = generate_subqueries(query, [], llm)  # Generate sub-queries
    
    if not is_coral_relevant:
        # Also check with Bayes filter as a secondary measure
        is_allowed, confidence = bayes_filter(query)  # Filter the query
        
        if not is_allowed:
            # Both LLM and Bayes filter indicate non-coral query
            print(f"\n{'='*40}\nFILTER ALERT: Query may be unrelated to coral research\n{'='*40}")
            
            warning_message = (
                "\n" + "="*40 + "\nATTENTION\n" + "="*40 + "\n"
                "This query appears unrelated to coral preservation and research. "
                "The CoralX Foundation created this tool specifically for coral research. "
                "Please use it wisely.\n\n"
                "Do you still want to proceed with this query? (yes/no): "
            )
            
            print(warning_message, end="")  # Prompt user for confirmation
            user_response = input().strip().lower()  # Get user response
            
            if user_response != "yes" and user_response != "y":
                log_unrelated_query(query, confidence, False)  # Log the unrelated query
                return (
                    "\n" + "="*40 + "\nQUERY CANCELLED\n" + "="*40 + "\n"
                    "Thank you for understanding. The CoralX Foundation is committed to "
                    "coral preservation and research. If you have coral-related questions, "
                    "please feel free to ask them anytime."
                )  # Return cancellation message
            
            log_unrelated_query(query, confidence, True)  # Log the query if user proceeds
            print("\nProceeding with query. Note that this instance will be reviewed.")
    
    # Create context-aware hybrid retriever
    retriever = ContextHybridRetriever(vectordb, graph, k=4, max_depth=2)  # Initialize retriever
    
    # Create QA chain with refine chain type
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="refine",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "document_variable_name": "context",
            "question_prompt": ChatPromptTemplate.from_template(
                "Based ONLY on the provided context, answer the question. If the answer cannot "
                "be fully determined from the context, say 'I don't have enough information to "
                "answer this question completely.'\n\n"
                "First provide a concise 1-paragraph answer with key facts from the context, "
                "then write 'Additional Details:' and add supporting information from the context.\n\n"
                "Context:\n{context}\n\nQuestion: {question}"
            ),
            "refine_prompt": ChatPromptTemplate.from_template(
                "Original question: {question}\n"
                "Current answer: {existing_answer}\n"
                "New context: {context}\n\n"
                "Based ONLY on the original context and new context:\n"
                "1. Keep the initial concise paragraph if supported by context\n"
                "2. Only add new information under 'Additional Details' if found in context\n"
                "3. If new context contradicts current answer, revise for accuracy\n"
                "4. If unsure or information is not in context, maintain uncertainty"
            )
        }
    )
    
    # Invoke the QA chain with the query
    try:
        llm_response = qa_chain.invoke({"query": query})  # Get response from QA chain
        return process_llm_response_with_sources(llm_response)  # Process and return the response
    except Exception as e:
        print(f"Error during query processing: {e}")  # Print error message
        return "An error occurred while processing your query. Please try again."  # Return error message

if __name__ == "__main__":
    query = "What is a drawback to lesion removal on diseased coral?"  # Example query
    result = answer_to_QA(query)  # Process the query
    print(result)  # Print the result 