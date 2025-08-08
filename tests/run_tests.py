import sys
import os

# Add the project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Run tests
if __name__ == "__main__":
    import pytest
    import sys
    
    # Run all tests
    exit_code = pytest.main(["-v", "tests/"])
    sys.exit(exit_code)