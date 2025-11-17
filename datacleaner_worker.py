import sys
from miditok import REMI
from pathlib import Path

def check_file(file_path_str):
    """
    Tries to tokenize a single file.
    Exits with 0 on success.
    Exits with 1 on a catchable Python error.
    (Will hard-crash on a C++ error, which is fine)
    """
    tokenizer = REMI()
    
    try:
        # Test-tokenize the file
        tokenizer(file_path_str)
        
    except Exception as e:
        # This catches "soft" corruption (Python errors)
        # We print the error so the main script can log it
        print(f"Python-level error: {e}", file=sys.stderr)
        sys.exit(1)
        
    # If it gets here, the file is valid
    sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python datacleaner_worker.py <path_to_file>", file=sys.stderr)
        sys.exit(1)
        
    file_path = sys.argv[1]
    check_file(file_path)