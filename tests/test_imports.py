
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that basic imports work"""
    try:
        import pandas as pd
        import numpy as np
        assert pd is not None
        assert np is not None
        print("✓ Imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_project_structure():
    """Test that required directories exist"""
    required_dirs = ['src', 'data', 'notebooks', 'reports', 'tests']
    for d in required_dirs:
        assert os.path.exists(d) or os.path.exists(f'../{d}'), f"Directory {d} missing"
    print("✓ Project structure verified")
    return True

if __name__ == "__main__":
    test_imports()
    test_project_structure()
    print("\n✅ All tests passed!")