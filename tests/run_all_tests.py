"""
Test runner for Medical Reasoning Dataset Generator.

This script runs all tests in the test suite and provides comprehensive
reporting for continuous integration and quality assurance.
"""

import sys
import unittest
import os
from pathlib import Path
import time
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from utils.logging_setup import setup_logging, get_logger
    logging_available = True
except ImportError:
    logging_available = False
    print("Warning: Logging setup not available, using basic print statements")


class ColoredTextTestResult(unittest.TextTestResult):
    """Enhanced test result with colored output and detailed reporting."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_results = []
        self.start_time = None
        
    def startTest(self, test):
        super().startTest(test)
        self.start_time = time.time()
        
    def addSuccess(self, test):
        super().addSuccess(test)
        duration = time.time() - self.start_time if self.start_time else 0
        self.test_results.append({
            'test': str(test),
            'status': 'PASS',
            'duration': duration,
            'message': None
        })
        if self.verbosity > 1:
            print(f"✓ {test} ({duration:.3f}s)")
    
    def addError(self, test, err):
        super().addError(test, err)
        duration = time.time() - self.start_time if self.start_time else 0
        self.test_results.append({
            'test': str(test),
            'status': 'ERROR',
            'duration': duration,
            'message': str(err[1])
        })
        if self.verbosity > 0:
            print(f"✗ {test} - ERROR ({duration:.3f}s)")
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        duration = time.time() - self.start_time if self.start_time else 0
        self.test_results.append({
            'test': str(test),
            'status': 'FAIL',
            'duration': duration,
            'message': str(err[1])
        })
        if self.verbosity > 0:
            print(f"✗ {test} - FAIL ({duration:.3f}s)")
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        duration = time.time() - self.start_time if self.start_time else 0
        self.test_results.append({
            'test': str(test),
            'status': 'SKIP',
            'duration': duration,
            'message': reason
        })
        if self.verbosity > 1:
            print(f"⊘ {test} - SKIPPED ({duration:.3f}s)")


class TestRunner:
    """Comprehensive test runner for the medical dataset generator."""
    
    def __init__(self, verbosity: int = 2):
        """
        Initialize test runner.
        
        Args:
            verbosity: Test output verbosity level (0-2)
        """
        self.verbosity = verbosity
        self.logger = None
        
        if logging_available:
            try:
                self.logger = setup_logging(
                    log_level="INFO",
                    log_file="logs/test_results.log",
                    console_output=False,
                    structured_logging=True
                )
            except Exception as e:
                print(f"Warning: Could not setup logging: {e}")
    
    def discover_tests(self, test_dir: str = None) -> unittest.TestSuite:
        """
        Discover all test files in the test directory.
        
        Args:
            test_dir: Directory to search for tests (default: current directory)
        
        Returns:
            Test suite containing all discovered tests
        """
        if test_dir is None:
            test_dir = Path(__file__).parent
        
        loader = unittest.TestLoader()
        
        # Discover all test files
        suite = loader.discover(
            start_dir=str(test_dir),
            pattern='test_*.py',
            top_level_dir=str(test_dir)
        )
        
        return suite
    
    def run_tests(self, test_suite: unittest.TestSuite = None) -> Dict[str, Any]:
        """
        Run all tests and return comprehensive results.
        
        Args:
            test_suite: Test suite to run (if None, discovers automatically)
        
        Returns:
            Dictionary containing test results and statistics
        """
        if test_suite is None:
            test_suite = self.discover_tests()
        
        print("=" * 70)
        print("Medical Reasoning Dataset Generator - Test Suite")
        print("=" * 70)
        
        # Custom test runner with enhanced reporting
        runner = unittest.TextTestRunner(
            verbosity=self.verbosity,
            resultclass=ColoredTextTestResult,
            stream=sys.stdout,
            buffer=True
        )
        
        start_time = time.time()
        result = runner.run(test_suite)
        total_time = time.time() - start_time
        
        # Compile results
        test_results = {
            'total_tests': result.testsRun,
            'passed': result.testsRun - len(result.failures) - len(result.errors),
            'failed': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'total_time': total_time,
            'success_rate': 0.0,
            'detailed_results': getattr(result, 'test_results', [])
        }
        
        if result.testsRun > 0:
            test_results['success_rate'] = (test_results['passed'] / result.testsRun) * 100
        
        # Print summary
        self._print_summary(test_results)
        
        # Log results if logging is available
        if self.logger:
            self._log_results(test_results)
        
        return test_results
    
    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print test results summary."""
        print("\n" + "=" * 70)
        print("TEST RESULTS SUMMARY")
        print("=" * 70)
        
        print(f"Total Tests:    {results['total_tests']}")
        print(f"Passed:         {results['passed']} ✓")
        print(f"Failed:         {results['failed']} ✗")
        print(f"Errors:         {results['errors']} ✗")
        print(f"Skipped:        {results['skipped']} ⊘")
        print(f"Success Rate:   {results['success_rate']:.1f}%")
        print(f"Total Time:     {results['total_time']:.2f}s")
        
        # Overall result
        if results['failed'] == 0 and results['errors'] == 0:
            print(f"\n🎉 ALL TESTS PASSED! 🎉")
            exit_code = 0
        else:
            print(f"\n❌ SOME TESTS FAILED")
            exit_code = 1
        
        print("=" * 70)
        
        # Set exit code for CI/CD
        sys.exit(exit_code)
    
    def _log_results(self, results: Dict[str, Any]) -> None:
        """Log test results for monitoring."""
        if not self.logger:
            return
        
        self.logger.info(
            "Test suite execution completed",
            extra={
                'total_tests': results['total_tests'],
                'passed': results['passed'],
                'failed': results['failed'],
                'errors': results['errors'],
                'skipped': results['skipped'],
                'success_rate': results['success_rate'],
                'total_time': results['total_time'],
                'operation': 'test_suite_execution'
            }
        )
        
        # Log individual test results
        for test_result in results.get('detailed_results', []):
            if test_result['status'] in ['FAIL', 'ERROR']:
                self.logger.error(
                    f"Test failed: {test_result['test']}",
                    extra={
                        'test_name': test_result['test'],
                        'status': test_result['status'],
                        'duration': test_result['duration'],
                        'error_message': test_result['message'],
                        'operation': 'individual_test_failure'
                    }
                )
    
    def run_specific_test(self, test_name: str) -> Dict[str, Any]:
        """
        Run a specific test by name.
        
        Args:
            test_name: Name of the test module or class to run
        
        Returns:
            Test results dictionary
        """
        loader = unittest.TestLoader()
        
        try:
            # Try to load as module
            suite = loader.loadTestsFromName(test_name)
        except (ImportError, AttributeError):
            print(f"Could not find test: {test_name}")
            return {'error': f'Test not found: {test_name}'}
        
        return self.run_tests(suite)
    
    def check_system_requirements(self) -> bool:
        """
        Check if system meets requirements for running tests.
        
        Returns:
            True if requirements are met, False otherwise
        """
        print("Checking system requirements...")
        
        requirements_met = True
        
        # Check Python version
        if sys.version_info < (3, 9):
            print("❌ Python 3.9+ required")
            requirements_met = False
        else:
            print("✓ Python version OK")
        
        # Check required packages
        required_packages = ['pydantic', 'yaml', 'pathlib']
        for package in required_packages:
            try:
                __import__(package)
                print(f"✓ {package} available")
            except ImportError:
                print(f"❌ {package} not available")
                requirements_met = False
        
        # Check test directory structure
        test_dir = Path(__file__).parent
        if not (test_dir / "test_basic_functionality.py").exists():
            print("❌ Basic test files not found")
            requirements_met = False
        else:
            print("✓ Test files found")
        
        return requirements_met


def main():
    """Main entry point for test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run tests for Medical Reasoning Dataset Generator"
    )
    parser.add_argument(
        '--test', '-t', 
        type=str, 
        help='Run specific test (module or class name)'
    )
    parser.add_argument(
        '--verbosity', '-v',
        type=int,
        choices=[0, 1, 2],
        default=2,
        help='Test output verbosity (0=quiet, 1=normal, 2=verbose)'
    )
    parser.add_argument(
        '--check-requirements',
        action='store_true',
        help='Check system requirements before running tests'
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner(verbosity=args.verbosity)
    
    # Check requirements if requested
    if args.check_requirements:
        if not runner.check_system_requirements():
            print("❌ System requirements not met")
            sys.exit(1)
        print("✓ All requirements met\n")
    
    # Run tests
    if args.test:
        # Run specific test
        results = runner.run_specific_test(args.test)
    else:
        # Run all tests
        results = runner.run_tests()
    
    # Results are handled by _print_summary which calls sys.exit()


if __name__ == "__main__":
    main()