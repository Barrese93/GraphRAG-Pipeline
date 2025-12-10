"""Test runner script for GraphRAG tests."""

import subprocess
import sys
import os

def run_tests():
    """Run all GraphRAG tests."""
    print("🚀 Running GraphRAG Test Suite")
    print("=" * 40)
    
    # Run pytest with coverage
    try:
        # Run all tests
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--tb=short",
            "--disable-warnings"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print(f"\nExit code: {result.returncode}")
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def run_unit_tests_only():
    """Run only unit tests (exclude integration tests)."""
    print("🧪 Running Unit Tests Only")
    print("=" * 30)
    
    try:
        # Run unit tests excluding integration tests
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/",
            "-v",
            "--tb=short",
            "--disable-warnings",
            "--ignore=tests/test_pipeline_integration.py"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print(f"\nExit code: {result.returncode}")
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running unit tests: {e}")
        return False

def run_specific_test(test_file):
    """Run a specific test file."""
    print(f"🎯 Running Specific Test: {test_file}")
    print("=" * 40)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            f"tests/{test_file}",
            "-v",
            "--tb=short",
            "--disable-warnings"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print(f"\nExit code: {result.returncode}")
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running specific test: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "unit":
            run_unit_tests_only()
        elif sys.argv[1] == "all":
            run_tests()
        else:
            run_specific_test(sys.argv[1])
    else:
        print("Usage:")
        print("  python tests/run_tests.py all      - Run all tests")
        print("  python tests/run_tests.py unit     - Run unit tests only")
        print("  python tests/run_tests.py <file>   - Run specific test file")
        print("\nAvailable test files:")
        for file in os.listdir("."):
            if file.startswith("test_") and file.endswith(".py"):
                print(f"  - {file}")
