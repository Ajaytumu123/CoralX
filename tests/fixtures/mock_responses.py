"""
Mock responses for testing the coral research system.

This module provides predefined responses to simulate various scenarios without
making actual API calls.
"""
from unittest.mock import MagicMock
from langchain.schema import Document

# Mock document data for testing
MOCK_DOCUMENTS = [
    Document(
        page_content="Coral bleaching occurs when corals expel their symbiotic algae due to stress, "
                    "primarily caused by elevated ocean temperatures. When water temperatures rise even "
                    "slightly above normal levels for an extended period, the symbiotic relationship "
                    "between corals and zooxanthellae algae breaks down. The algae provide corals with "
                    "nutrients and their characteristic colors, so their loss leaves the corals looking "
                    "white or 'bleached'. While bleached corals aren't dead, they're severely stressed "
                    "and more susceptible to mortality if conditions don't improve.",
        metadata={
            "source": "coral_bleaching_research.pdf",
            "page": 12,
            "author": "Marine Biology Institute"
        }
    ),
    Document(
        page_content="Ocean acidification poses a significant threat to coral reefs worldwide. As carbon "
                    "dioxide (CO2) levels rise in the atmosphere, the ocean absorbs approximately 30% of it, "
                    "leading to chemical reactions that decrease the pH of seawater. This acidification "
                    "reduces the availability of carbonate ions, making it increasingly difficult for "
                    "corals to build their calcium carbonate skeletons. Research shows that under high-CO2 "
                    "conditions, coral calcification rates decrease by as much as 40%, leading to weaker "
                    "reef structures and reduced growth rates.",
        metadata={
            "source": "ocean_acidification_impacts.pdf",
            "page": 45,
            "author": "Climate Research Center"
        }
    ),
    Document(
        page_content="Coral restoration programs have shown promising results in helping degraded reef "
                    "systems recover. These programs typically involve techniques such as coral gardening, "
                    "where fragments of healthy corals are grown in nurseries before being transplanted "
                    "to damaged reef areas. The success rates vary depending on species, location, and "
                    "local conditions, but some projects have achieved coral survival rates of over 70%. "
                    "Additional approaches include larval seeding and creating artificial reef structures. "
                    "While restoration isn't a substitute for addressing the root causes of coral decline, "
                    "it serves as a valuable tool for rehabilitating critically damaged ecosystems.",
        metadata={
            "source": "restoration_techniques.pdf",
            "page": 78,
            "author": "Reef Rehabilitation Foundation"
        }
    ),
]

# Mock LLM response for coral-related queries
MOCK_CORAL_RESPONSE = {
    'result': (
        "Coral bleaching is primarily caused by rising ocean temperatures associated with climate change. "
        "When water temperatures rise above normal levels for extended periods, corals expel their "
        "symbiotic algae (zooxanthellae), resulting in the white or 'bleached' appearance. "
        "While bleached corals aren't dead, they're under severe stress and face higher mortality risks "
        "if conditions don't improve quickly.\n\n"
        "Additional Details:\n"
        "Other factors contributing to coral bleaching include ocean acidification from increased "
        "atmospheric CO2, pollution, excessive sunlight, and exposure to air during extreme low tides. "
        "Major global bleaching events have become more frequent, with significant events recorded in "
        "1998, 2010, and 2014-2017. The Great Barrier Reef has experienced several severe bleaching "
        "events, with over 50% of its corals lost since 1995 according to some studies. "
        "While some coral species show more resilience than others, continued warming trends "
        "threaten even the most adaptive coral populations."
    ),
    'source_documents': MOCK_DOCUMENTS
}

# Mock LLM response for non-coral queries (when approved)
MOCK_GENERAL_RESPONSE = {
    'result': (
        "I've searched for information about this topic, but it appears to be outside the scope of "
        "our coral research database. While I can provide general information based on my training, "
        "I don't have specialized sources on this particular subject.\n\n"
        "Additional Details:\n"
        "For the most accurate and up-to-date information on this topic, I'd recommend consulting "
        "resources specifically focused on this area. The CoralX Foundation specializes in coral "
        "research and preservation, so our knowledge base is optimized for queries related to "
        "marine ecosystems, particularly coral reefs and their conservation."
    ),
    'source_documents': []
}

def get_mock_retriever(document_list=None):
    """Create a mock retriever that returns predefined documents"""
    
    if document_list is None:
        document_list = MOCK_DOCUMENTS
        
    mock_retriever = MagicMock()
    mock_retriever.get_relevant_documents.return_value = document_list
    
    return mock_retriever

def get_mock_vectorstore():
    """Create a mock vector store that returns a predefined retriever"""
    
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = get_mock_retriever()
    
    return mock_vectorstore 