#!/usr/bin/env python3
"""
Script to demonstrate how to run tests with coverage reporting
"""

import subprocess
import sys
import os

def run_tests_with_coverage():
    """Run tests with coverage reporting"""
    try:
        # Change to the project directory
        project_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(project_dir)
        
        # Run pytest with coverage
        cmd = [
            sys.executable, "-m", "pytest",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "-v",
            "tests/"
        ]
        
        print("Running tests with coverage...")
        print(f"Command: {' '.join(cmd)}")
        print("--------------------------------------------------")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        print(f"Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("\n Tests completed successfully!")
            print(" Coverage report generated in htmlcov/ directory")
        else:
            print("\n Tests failed!")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

if __name__ == "__main__":
    print("Test Runner with Coverage Report")
    print("========================================")
    
    success = run_tests_with_coverage()
    
    if success:
        print("\n Test execution completed!")
    else:
        print("\n Test execution failed!")
        
    sys.exit(0 if success else 1)