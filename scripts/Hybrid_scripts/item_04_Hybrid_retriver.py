# #### RUN: python scripts/Hybrid-scripts/item_04_Hybrid_retriver.py

import os
import re
import sys
import urllib.parse
import time
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.retrievers import BaseRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.globals import set_verbose, set_debug
from item_01_database_creation_Hybrid import NetworkXGraph
from pydantic import SkipValidation

# Disable verbose and debug logging
set_verbose(False)
set_debug(False)

# Load environment variables from a .env file
load_dotenv(find_dotenv())

# Constants
LLM_MODEL = "gpt-4o-mini"  # Model used for language processing
VECTOR_STORE_PATH = "hybrid_index"  # Path to the FAISS vector store
GRAPH_PATH = "hybrid_index/networkx_graph.pkl"  # Path to the NetworkX graph

print("=" * 50)
print("Coral Reef Hybrid Retriever")
print("=" * 50)

class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that combines vector search with graph-based retrieval.

    This class is responsible for retrieving relevant documents by leveraging both
    vector search and graph-based search techniques. It enhances the retrieval process
    by boosting the relevance of documents that are found in both searches.

    Attributes:
        vector_retriever (SkipValidation[any]): The vector store retriever instance.
        graph (SkipValidation[any]): The graph instance used for graph-based retrieval.
        k (int): The number of results to retrieve from vector search.
    """
    vector_retriever: SkipValidation[any] = None
    graph: SkipValidation[any] = None
    k: int = 5  # Default number of results to retrieve

    def __init__(self, vector_store, graph, k=5):
        """
        Initialize the hybrid retriever with a vector store and a graph.

        Args:
            vector_store: FAISS vector store instance for vector retrieval.
            graph: NetworkXGraph object for graph-based retrieval.
            k: Number of results to retrieve from vector search (default is 5).
        """
        super().__init__()  # Initialize the base class
        self.vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})  # Set up vector retriever
        self.graph = graph  # Assign the graph for retrieval
        self.k = k  # Set the number of results to retrieve

    def _get_relevant_documents(self, query: str):
        """
        Get relevant documents using both vector search and graph-based retrieval.

        This method performs a hybrid search by first retrieving documents from the
        vector store and then enriching the results with graph-based search results.
        It boosts the relevance of documents found in both searches.

        Args:
            query: Query string to search for relevant documents.

        Returns:
            List of relevant documents enriched with metadata indicating their source
            and relevance score.

        Raises:
            Exception: If an error occurs during retrieval, it falls back to vector search results.
        """
        try:
            # Get vector search results
            print(f"Performing vector search for query: {query[:50]}...")  # Log the query
            vector_docs = self.vector_retriever.invoke(query)  # Retrieve documents from vector store
            print(f"Retrieved {len(vector_docs)} documents from vector search")  # Log the number of documents retrieved
            
            # Get graph search results
            print(f"Performing graph search for query: {query[:50]}...")  # Log the query
            graph_results = self.graph.query_graph(query)  # Retrieve documents from graph
            print(f"Retrieved {len(graph_results)} document references from graph search")  # Log the number of documents retrieved
            
            # Extract document IDs from graph results
            graph_doc_ids = {r["id"] for r in graph_results}  # Create a set of document IDs from graph results
            
            # Boost documents found in both searches
            hybrid_docs = []  # List to hold hybrid documents
            for doc in vector_docs:
                if doc.metadata.get("doc_id") in graph_doc_ids:  # Check if the document is in graph results
                    doc.metadata["graph_enriched"] = True  # Mark document as enriched
                    # Add a relevance boost to documents found in both searches
                    doc.metadata["relevance_boost"] = graph_results[list(graph_doc_ids).index(doc.metadata["doc_id"])].get("relevance_score", 1)
                hybrid_docs.append(doc)  # Add document to hybrid list
            
            # Sort by relevance if graph-enriched
            hybrid_docs.sort(key=lambda x: x.metadata.get("relevance_boost", 0), reverse=True)  # Sort documents by relevance
            
            return hybrid_docs  # Return the enriched list of documents
            
        except Exception as e:
            print(f"Error in hybrid retrieval: {e}")  # Log the error
            # Fallback to vector search if graph search fails
            return vector_docs if 'vector_docs' in locals() else []  # Return vector docs if available

# Function to generate structured CoT prompt
def generate_cot_prompt():
    """
    Generate Chain-of-Thought (CoT) prompt for information retrieval.

    This function creates a structured prompt that guides the language model in
    analyzing context information and answering questions based on that context.

    Returns:
        str: A formatted prompt string for the language model.
    """
    return """
    You are a specialized retrieval system analyzing scientific information.

    CONTEXT INFORMATION:
    {context}

    QUESTION: {question}

    REASONING INSTRUCTIONS:
    1. First, evaluate each piece of context for relevance to the question (score 1-5)
    2. For relevant information, extract key facts with direct citations
    3. Identify any knowledge gaps or contradictions between sources
    4. Synthesize the information into a coherent understanding
    5. Apply logical reasoning to reach a conclusion

    RESPONSE FORMAT:
    [Relevance Analysis]
    - Analysis of which context pieces are most useful and why

    [Key Information]
    - Extracted information with source citations

    [Reasoning Process]
    - Step-by-step logical process connecting facts to answer

    [Answer]
    - Clear, concise answer based only on provided context
    - If the answer cannot be determined from context, state: "The provided information is insufficient to answer this question."

    IMPORTANT:
    - Do not introduce facts not present in the context
    - Do not use uncertain language when information is clearly stated
    - Be precise about what information comes from which source
    - Do not prefix your answer with phrases like "based on the context" or "the information suggests"
    """  # Return the structured prompt

def generate_old_cot_prompt():
    """
    Generate an older style Chain-of-Thought (CoT) prompt for information retrieval.

    This function creates a simpler prompt format for the language model to follow
    when answering questions based on provided context.

    Returns:
        str: A formatted prompt string for the language model.
    """
    return """
        You are a helpful assistant that answers questions. 
        Respond only to the question asked, response should be relevant to the question.
        If the answer cannot be deduced from the context, do not give an answer just say "Not Relevant to the context, sorry i cannot answer this question".
        Question should be answered from the given context.
        Use the following context to answer the question and You will use a Chain of Thought (CoT) reasoning process and final answer should be in last paragraph.
        Avoid prefacing your final answer with phrases such as "we look at the data provided in the context", "The context states that","Therefore, the final answer is:" ,"In summary", "thus" or "In conclusion," in the final response.
        Context: {context}

        Question: {question}

        Think step-by-step and then provide the final answer.
    """  # Return the old style prompt

# Function to clean the final answer
def clean_final_answer(final_answer):
    """
    Clean up the final answer by removing unnecessary tags and formatting.

    This function processes the final answer string to ensure it is clean and
    free of any extraneous formatting or prefixes that may confuse the reader.

    Args:
        final_answer: The raw final answer string from the language model.

    Returns:
        str: A cleaned version of the final answer.
    """
    try:
        final_answer = final_answer.strip()  # Remove leading and trailing whitespace
        # Remove various "Final Answer:" prefixes
        final_answer = re.sub(r'\d+\.\s*\*\*Final Answer\*\*:\s*|\*\*Final Answer\*\*: |Final Answer:\s* |Final answer:\s\s*', '', final_answer, flags=re.IGNORECASE)
        # Remove markdown formatting
        final_answer = re.sub(r'\*\*', '', final_answer)  # Remove bold formatting
        return final_answer  # Return the cleaned answer
    except Exception as e:
        print(f"Error cleaning final answer: {e}")  # Log the error
        return final_answer if 'final_answer' in locals() else "Error processing answer"  # Return error message if cleaning fails

# Function to process LLM response and extract final answer, CoT, and citations
def process_llm_response_with_sources(llm_response):
    """
    Process the LLM response to extract the final answer, reasoning steps, and citations.

    This function analyzes the response from the language model, extracting
    relevant information such as the final answer, reasoning steps, and citations
    for the provided context.

    Args:
        llm_response: Response from the language model, expected to be a dictionary.

    Returns:
        Tuple of (final_answer, reasoning_steps, citations, full_response):
            - final_answer (str): The final answer extracted from the response.
            - reasoning_steps (str): The reasoning steps provided by the model.
            - citations (list): List of citations extracted from the response.
            - full_response (str): The complete raw response from the model.
    """
    try:
        full_response = llm_response['result'].strip()  # Get the raw response and strip whitespace
        
        # Check if the response indicates that no answer could be found
        result_lower = full_response.lower().split(".")[0]  # Get the first sentence for analysis
        irrelevant_phrases = [
            "i don't know", "i do not know", "i'm not sure", "i am not sure",
            "not relevant to the context", "sorry i cannot answer this question",
            "the provided information is insufficient"
        ]  # List of phrases indicating irrelevance
        is_irrelevant = any(phrase in result_lower for phrase in irrelevant_phrases)  # Check for irrelevance

        if is_irrelevant:
            return full_response, "No reasoning steps provided.", [], full_response  # Return early if irrelevant

        # Extract citations
        citations = []  # List to hold citations
        unique_citations = set()  # Set to track unique citations
        for source in llm_response.get("source_documents", []):  # Iterate over source documents
            try:
                citation = source.metadata.get('source', 'Unknown source')  # Get the citation source
                if citation not in unique_citations and citation != 'Unknown source':  # Check for uniqueness
                    unique_citations.add(citation)  # Add to unique set
                    # Create a clickable Google search link for the citation
                    encoded_citation = urllib.parse.quote_plus(citation)  # URL-encode the citation
                    google_search_link = f"https://www.google.com/search?q={encoded_citation}"  # Create search link
                    clickable_link = f'<a href="{google_search_link}" target="_blank">{citation}</a>'  # Format as HTML link
                    citations.append(clickable_link)  # Add to citations list
            except Exception as e:
                print(f"Error processing citation: {e}")  # Log citation processing errors

        # Extract reasoning steps and final answer
        if "[Answer]" in full_response:
            # For structured format responses
            parts = full_response.split("[Answer]")  # Split by the answer marker
            reasoning_steps = parts[0].strip()  # Get reasoning steps
            final_answer = parts[1].strip() if len(parts) > 1 else full_response  # Get final answer
        elif "Final Answer:" in full_response:
            # For traditional CoT responses
            reasoning_steps = full_response.split("Final Answer:")[0].strip()  # Get reasoning steps
            final_answer = full_response.split("Final Answer:")[1].strip()  # Get final answer
        else:
            # For unstructured responses, assume the last paragraph is the answer
            paragraphs = full_response.split("\n\n")  # Split by double newlines
            if len(paragraphs) > 1:
                reasoning_steps = "\n\n".join(paragraphs[:-1]).strip()  # Get all but the last paragraph as reasoning
                final_answer = paragraphs[-1].strip()  # Last paragraph as final answer
            else:
                reasoning_steps = "No explicit reasoning steps provided."  # Default message for reasoning
                final_answer = full_response.strip()  # Use full response as final answer

        # Clean the final answer
        final_answer = clean_final_answer(final_answer)  # Clean the final answer
        
        return final_answer, reasoning_steps, citations, full_response  # Return extracted information
        
    except Exception as e:
        print(f"Error processing LLM response: {e}")  # Log processing errors
        return "Error processing response", "Error in processing", [], "Error in processing"  # Return error messages

# Function to get raw LLM response with CoT
def raw_LLM_response(query, retry_attempts=2):
    """
    Get raw LLM response with Chain-of-Thought reasoning.

    This function interacts with the language model to retrieve a response
    for a given query, implementing retry logic in case of failures.

    Args:
        query: Query string to be processed by the language model.
        retry_attempts: Number of retry attempts for API calls (default is 2).

    Returns:
        dict: Raw LLM response containing the result and source documents.
    """
    for attempt in range(retry_attempts + 1):  # Loop for retry attempts
        try:
            print(f"Attempt {attempt + 1}/{retry_attempts + 1} to get response for query: {query[:50]}...")  # Log attempt
            # Initialize OpenAI embeddings
            embedding = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))  # Load embeddings with API key
            
            # Load the vector store
            print(f"Loading vector store from {VECTOR_STORE_PATH}...")  # Log loading vector store
            vectordb = FAISS.load_local(VECTOR_STORE_PATH, embedding, allow_dangerous_deserialization=True)  # Load vector store
            
            # Load the graph
            print(f"Loading graph from {GRAPH_PATH}...")  # Log loading graph
            graph = NetworkXGraph.load_graph(GRAPH_PATH)  # Load graph
            
            # Create Hybrid Retriever
            print("Creating hybrid retriever...")  # Log retriever creation
            hybrid_retriever = HybridRetriever(vectordb, graph)  # Instantiate hybrid retriever
            
            # Initialize the LLM
            print(f"Initializing LLM with model {LLM_MODEL}...")  # Log LLM initialization
            llm = ChatOpenAI(model=LLM_MODEL, temperature=0)  # Create LLM instance
            
            # Create prompt template
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template=generate_cot_prompt()  # Use the structured CoT prompt
            )
            
            # Create QA chain
            print("Creating QA chain...")  # Log QA chain creation
            qa_chain = RetrievalQA.from_chain_type(
                llm,
                chain_type="stuff",
                retriever=hybrid_retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt_template}  # Pass the prompt template
            )
            
            # Get response
            print("Getting response from LLM...")  # Log response retrieval
            start_time = time.time()  # Start timing
            # Use invoke instead of __call__ to avoid deprecation warnings
            llm_response = qa_chain.invoke({"query": query})  # Invoke the QA chain with the query
            end_time = time.time()  # End timing
            print(f"LLM response received in {end_time - start_time:.2f} seconds")  # Log response time
            
            return llm_response  # Return the LLM response
            
        except Exception as e:
            print(f"Error in attempt {attempt + 1}: {e}")  # Log the error
            if attempt < retry_attempts:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")  # Log retry wait time
                time.sleep(wait_time)  # Wait before retrying
            else:
                print("All retry attempts failed.")  # Log failure
                return {"result": "Error: Unable to get response from LLM", "source_documents": []}  # Return error message

# Function to answer QA with CoT
def answer_to_QA(query):
    """
    Answer a question using Chain-of-Thought reasoning.

    This function processes a query by retrieving a response from the language model
    and extracting the final answer, reasoning steps, and citations.

    Args:
        query: Query string to be answered.

    Returns:
        Tuple of (final_answer, reasoning_steps, citations, full_response):
            - final_answer (str): The final answer extracted from the response.
            - reasoning_steps (str): The reasoning steps provided by the model.
            - citations (list): List of citations extracted from the response.
            - full_response (str): The complete raw response from the model.
    """
    print(f"\nProcessing query: {query}")  # Log the query being processed
    
    # Get raw LLM response
    llm_response = raw_LLM_response(query)  # Retrieve response from LLM
    
    # Process LLM response
    final_answer, reasoning_steps, citations, full_response = process_llm_response_with_sources(llm_response)  # Extract information
    
    return final_answer, reasoning_steps, citations, full_response  # Return the results

# Main Function
if __name__ == "__main__":
    # Get query from command line arguments or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])  # Join command line arguments as query
    else:
        # Default query
        #query = "What is a drawback to lesion removal on diseased coral?"
        query = "What are the primary threats to the breeding populations of albatrosses in the NWHI?"  # Default query
        print(f"No query provided. Using default query: {query}")  # Log default query
    
    # Get answer
    final_answer, reasoning_steps, citations, full_response = answer_to_QA(query)  # Retrieve answer
    
    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    print("\nFinal Answer:")
    print(final_answer)  # Print the final answer
    
    print("\nChain of Thoughts:")
    print(reasoning_steps)  # Print reasoning steps
    
    if citations:
        print("\nCitations:")
        for citation in citations:
            print(citation)  # Print each citation
    else:
        print("\nNo citations found.")  # Log absence of citations
    
    # Write results to file
    try:
        os.makedirs("results", exist_ok=True)  # Create results directory if it doesn't exist
        output_file = os.path.join("results", f"query_result_{time.strftime('%Y%m%d_%H%M%S')}.txt")  # Define output file path
        with open(output_file, "w") as f:
            f.write(f"Query: {query}\n\n")  # Write the query to the file
            f.write(f"Final Answer:\n{final_answer}\n\n")  # Write the final answer to the file
            f.write(f"Chain of Thoughts:\n{reasoning_steps}\n\n")  # Write reasoning steps to the file
            f.write("Citations:\n")  # Write citations header
            for citation in citations:
                f.write(f"{citation.replace('<a href=', '').replace('</a>', '').split('>')[1]}\n")  # Write each citation
        print(f"\nResults saved to {output_file}")  # Log successful save
    except Exception as e:
        print(f"Error saving results to file: {e}")  # Log file save errors


