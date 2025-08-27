#!/usr/bin/env python3
"""
Verification script to check that all files have been cleaned of emoticons and hashtags
"""

import os
import re

def check_file_for_emoticons_and_hashtags(file_path):
    """Check if a file contains emoticons or hashtags"""
    emoticon_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]')
    hashtag_pattern = re.compile(r'#\w+')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        emoticons_found = emoticon_pattern.findall(content)
        hashtags_found = hashtag_pattern.findall(content)
        
        if emoticons_found or hashtags_found:
            print(f"File {file_path} contains:")
            if emoticons_found:
                print(f"  Emoticons: {emoticons_found}")
            if hashtags_found:
                print(f"  Hashtags: {hashtags_found}")
            return False
        else:
            print(f"File {file_path} is clean")
            return True
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

def main():
    """Main function to check all relevant files"""
    files_to_check = [
        'demo_run.py',
        'TESTING_UPGRADES_SUMMARY.md',
        'TESTING.md',
        'README.md',
        'verify_installation.py',
        'run_sample_test.py',
        'run_tests_with_coverage.py'
    ]
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    all_clean = True
    
    print("Checking files for emoticons and hashtags...")
    print("=" * 50)
    
    for file_name in files_to_check:
        file_path = os.path.join(project_dir, file_name)
        if os.path.exists(file_path):
            if not check_file_for_emoticons_and_hashtags(file_path):
                all_clean = False
        else:
            print(f"File not found: {file_path}")
    
    print("\n" + "=" * 50)
    if all_clean:
        print("All files are clean of emoticons and hashtags!")
    else:
        print("Some files still contain emoticons or hashtags.")
    
    return all_clean

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)