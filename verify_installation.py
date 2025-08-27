#!/usr/bin/env python3
"""
Verification script to check if all required dependencies are installed
"""

def check_dependencies():
    """Check if all required dependencies are installed"""
    dependencies = [
        'torch',
        'transformers',
        'pandas',
        'numpy',
        'sklearn',
        'pytest',
        'hypothesis'
    ]
    
    missing = []
    
    for dep in dependencies:
        try:
            if dep == 'sklearn':
                import sklearn
            else:
                __import__(dep)
            print(f" {dep} is installed")
        except ImportError:
            print(f" {dep} is missing")
            missing.append(dep)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    else:
        print("\nAll dependencies are installed!")
        return True

if __name__ == "__main__":
    check_dependencies()