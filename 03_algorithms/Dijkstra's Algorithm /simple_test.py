"""
Simple test file to verify Python execution
"""
import os
import sys

print("=== SIMPLE TEST ===")
print("This is a test message")
print("Current directory:", os.getcwd())
print("Files:", os.listdir('.'))
print("Python path:", sys.path)

# Try writing to a test file
try:
    with open('test_output.txt', 'w') as f:
        f.write("Test successful\n")
    print("Successfully wrote test file")
except Exception as e:
    print("Failed to write test file:", e)

print("Test complete")
