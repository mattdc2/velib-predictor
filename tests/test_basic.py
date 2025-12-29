"""Basic tests to ensure CI works."""


def test_imports():
    """Test that core modules can be imported."""
    from src.data.collector import VelibAPIClient
    from src.data.database import DatabaseManager

    assert DatabaseManager is not None
    assert VelibAPIClient is not None


def test_placeholder():
    """Placeholder test."""
    assert 1 + 1 == 2
