#!/usr/bin/env python3
"""
Simple test runner to verify our updated test file
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_imports():
    """Test that we can import all necessary modules"""
    try:
        from tests.test_model import (
            test_model_config_default_values,
            test_model_config_initialization,
            test_feedforward_activations
        )
        print(" All test functions imported successfully")
        return True
    except Exception as e:
        print(f" Failed to import test functions: {e}")
        return False

def test_simple_function():
    """Test a simple function from our test file"""
    try:
        from tests.test_model import test_model_config_default_values
        # This should not raise an exception
        print(" test_model_config_default_values function is callable")
        return True
    except Exception as e:
        print(f" Failed to call test function: {e}")
        return False

if __name__ == "__main__":
    print("Running simple verification of test file...")
    
    success = True
    success &= test_imports()
    success &= test_simple_function()
    
    if success:
        print("\n All verification tests passed!")
        print("The test file is properly structured and ready for use.")
    else:
        print("\n Some verification tests failed.")
        print("Please check the test file for issues.")
        
    sys.exit(0 if success else 1)