# CoralAI Test Suite

This directory contains the testing framework for the CoralAI project, which includes a comprehensive suite of unit tests, integration tests, and system tests for verifying the functionality of the coral research query system.

## Test Structure

The test suite is organized as follows:

```
tests/
├── fixtures/             # Test fixtures and mock data
│   ├── mock_responses.py # Mock responses for testing
│   └── sample_queries.py # Sample queries for different test scenarios
├── unit/                 # Unit tests for individual components
│   └── test_bayes_filter.py  # Tests for the Bayes filter functionality
├── integration/          # Integration tests for component interactions
│   └── test_query_processing.py  # Tests for the query processing flow
├── system/               # End-to-end system tests
│   └── test_end_to_end.py  # Full system tests
├── mocks/                # Additional mock objects and utilities
├── conftest.py           # Pytest configuration and fixtures
└── README.md             # This file
```

## Types of Tests

### Unit Tests
Unit tests focus on testing individual components in isolation to ensure they function correctly. These tests use mocks to isolate the component being tested.

### Integration Tests
Integration tests verify that multiple components work correctly together. These tests validate the interactions between different parts of the system.

### System Tests
System tests validate the entire system by running end-to-end scenarios that mimic real-world usage. These tests ensure that all components work together correctly.

## Test Fixtures

The `fixtures` directory contains:

- `sample_queries.py`: Collections of queries for different testing scenarios (coral-related, irrelevant, ambiguous, etc.)
- `mock_responses.py`: Mock responses to simulate various scenarios without making actual API calls

## Running Tests

### All Tests

To run all tests:

```bash
pytest
```

### Specific Test Categories

To run specific categories of tests:

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run only system tests
pytest tests/system/

# Run tests with specific markers
pytest -m "not slow"  # Skip slow tests
pytest -m "integration"  # Run only integration tests
```

### Test Coverage

To generate a test coverage report:

```bash
pytest --cov=scripts
```

## Test Markers

The following markers are available:

- `slow`: Tests that take a long time to run
- `integration`: Integration tests
- `system`: System tests
- `model`: Tests requiring the actual model file
- `openai`: Tests requiring OpenAI API access

## Adding New Tests

When adding new tests:

1. Place unit tests in the `unit` directory
2. Place integration tests in the `integration` directory
3. Place system tests in the `system` directory
4. Use appropriate fixtures from `conftest.py`
5. Add test markers to indicate test type and requirements

## Continuous Integration

The test suite is designed to run in a CI environment. The following considerations apply:

- Mock external dependencies to avoid API calls in CI
- Use markers to control which tests run in different environments
- System tests requiring full dependencies should be marked appropriately

## Troubleshooting

If tests are failing:

1. Check that all dependencies are installed: `pip install -r requirements.txt`
2. Verify that the model file is available in the expected location
3. Check environment variables for API keys if needed
4. Ensure tests are run from the project root directory 