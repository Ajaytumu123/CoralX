"""
Sample queries for testing the coral research system.

This module provides collections of queries for different testing scenarios.
"""

# Coral-related queries that should pass the filter
CORAL_QUERIES = [
    "What causes coral bleaching?",
    "How do coral reefs form?",
    "What is the impact of climate change on coral reefs?",
    "How can we protect endangered coral species?",
    "What is coral restoration?",
    "How do corals reproduce?",
    "What symbiotic relationships exist in coral ecosystems?",
    "What are the main threats to coral reefs today?",
    "How does ocean acidification affect coral growth?",
    "What role do corals play in marine biodiversity?"
]

# Queries that are clearly unrelated to coral research
IRRELEVANT_QUERIES = [
    "What is the best chocolate cake recipe?",
    "How to fix a flat tire?",
    "Latest smartphone reviews",
    "Who won the Super Bowl last year?",
    "How to train for a marathon",
    "Best practices for growing tomatoes",
    "History of jazz music",
    "How to make homemade pasta",
    "Investment strategies for beginners",
    "Tips for learning a new language"
]

# Edge case queries that are ambiguous or partially related
AMBIGUOUS_QUERIES = [
    "What fish live in the ocean?",  # Ocean-related but not coral-specific
    "Ocean temperature changes",  # Related to coral conditions but indirectly
    "Marine biology career paths",  # Field related to coral study but not directly about coral
    "Scuba diving techniques",  # Activity around reefs but not about coral
    "Sustainable fishing practices",  # Affects reef ecosystems but not coral-specific
    "What causes red tides?",  # Marine phenomenon but not coral-specific
    "How to identify tropical fish",  # Related to reef environments but not coral
    "Seashell collecting guide",  # Marine-related but not coral research
    "Underwater photography tips",  # Could be used for coral documentation but not specific
    "Marine conservation initiatives"  # May include coral conservation but broader
]

# Test queries with special characters or formatting
SPECIAL_FORMAT_QUERIES = [
    "coral bleaching?!?",
    "CORAL REEF PROTECTION",
    "   how to save corals   ",
    "coral + climate change + ocean",
    "coral; reef; conservation",
    "what-is-coral-spawning",
    "coral\nreef\necosystems",
    "'coral reef' AND 'biodiversity'",
    "coral reef (endangered species)",
    "https://example.com/coral-research"
]

def get_all_test_queries():
    """Return all test queries combined into a single list with labels"""
    all_queries = []
    
    for query in CORAL_QUERIES:
        all_queries.append({"query": query, "category": "relevant"})
    
    for query in IRRELEVANT_QUERIES:
        all_queries.append({"query": query, "category": "irrelevant"})
    
    for query in AMBIGUOUS_QUERIES:
        all_queries.append({"query": query, "category": "ambiguous"})
    
    for query in SPECIAL_FORMAT_QUERIES:
        all_queries.append({"query": query, "category": "special_format"})
    
    return all_queries 