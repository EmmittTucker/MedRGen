"""
Professional logging setup for medical dataset generation.

This module provides comprehensive logging functionality with configurable
levels, file output, console output, and structured logging for production
and commercial use.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'case_id'):
            log_data['case_id'] = record.case_id
        if hasattr(record, 'patient_id'):
            log_data['patient_id'] = record.patient_id
        if hasattr(record, 'doctor_id'):
            log_data['doctor_id'] = record.doctor_id
        if hasattr(record, 'operation'):
            log_data['operation'] = record.operation
        if hasattr(record, 'duration'):
            log_data['duration_seconds'] = record.duration
        if hasattr(record, 'error_type'):
            log_data['error_type'] = record.error_type
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


class MedicalDatasetLogger:
    """
    Professional logging system for medical dataset generation.
    
    Provides structured logging with multiple handlers, configurable levels,
    and specific logging utilities for medical case generation workflows.
    """
    
    def __init__(
        self,
        name: str = "medical_dataset_generator",
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        console_output: bool = True,
        structured_logging: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """
        Initialize the logging system.
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional)
            console_output: Whether to output to console
            structured_logging: Whether to use structured JSON logging
            max_file_size: Maximum file size before rotation
            backup_count: Number of backup files to keep
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        if console_output:
            self._setup_console_handler(structured_logging)
        
        if log_file:
            self._setup_file_handler(log_file, structured_logging, max_file_size, backup_count)
        
        # Setup error file handler for errors and critical messages
        if log_file:
            error_file = log_file.replace('.log', '_errors.log')
            self._setup_error_file_handler(error_file, structured_logging, max_file_size, backup_count)
    
    def _setup_console_handler(self, structured: bool) -> None:
        """Setup console output handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        
        if structured:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(
        self, 
        log_file: str, 
        structured: bool, 
        max_size: int, 
        backup_count: int
    ) -> None:
        """Setup rotating file handler."""
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        if structured:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
            )
        
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def _setup_error_file_handler(
        self, 
        error_file: str, 
        structured: bool, 
        max_size: int, 
        backup_count: int
    ) -> None:
        """Setup separate error file handler."""
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        
        if structured:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
            )
        
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        return self.logger
    
    # Convenience methods for medical dataset specific logging
    def log_case_generation_start(self, case_id: str, specialty: str, complexity: str) -> None:
        """Log the start of medical case generation."""
        self.logger.info(
            "Starting medical case generation",
            extra={
                'case_id': case_id,
                'specialty': specialty,
                'complexity': complexity,
                'operation': 'case_generation_start'
            }
        )
    
    def log_case_generation_complete(self, case_id: str, duration: float, quality_score: Optional[float] = None) -> None:
        """Log successful completion of medical case generation."""
        extra_data = {
            'case_id': case_id,
            'duration': duration,
            'operation': 'case_generation_complete'
        }
        if quality_score is not None:
            extra_data['quality_score'] = quality_score
        
        self.logger.info(
            "Medical case generation completed successfully",
            extra=extra_data
        )
    
    def log_case_generation_error(self, case_id: str, error: Exception, duration: float) -> None:
        """Log error during medical case generation."""
        self.logger.error(
            f"Medical case generation failed: {str(error)}",
            extra={
                'case_id': case_id,
                'duration': duration,
                'error_type': type(error).__name__,
                'operation': 'case_generation_error'
            },
            exc_info=True
        )
    
    def log_openai_api_call(self, model: str, tokens_used: int, cost: Optional[float] = None) -> None:
        """Log OpenAI API usage."""
        extra_data = {
            'model': model,
            'tokens_used': tokens_used,
            'operation': 'openai_api_call'
        }
        if cost is not None:
            extra_data['cost_usd'] = cost
        
        self.logger.debug(
            f"OpenAI API call completed - Model: {model}, Tokens: {tokens_used}",
            extra=extra_data
        )
    
    def log_dataset_evaluation(self, case_id: str, medical_accuracy: float, conversation_quality: float) -> None:
        """Log dataset evaluation results."""
        self.logger.info(
            "Dataset evaluation completed",
            extra={
                'case_id': case_id,
                'medical_accuracy_score': medical_accuracy,
                'conversation_quality_score': conversation_quality,
                'operation': 'dataset_evaluation'
            }
        )
    
    def log_batch_processing(self, batch_size: int, completed: int, errors: int, duration: float) -> None:
        """Log batch processing results."""
        self.logger.info(
            f"Batch processing completed - {completed}/{batch_size} successful, {errors} errors",
            extra={
                'batch_size': batch_size,
                'completed_cases': completed,
                'error_count': errors,
                'duration': duration,
                'operation': 'batch_processing'
            }
        )


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    structured_logging: bool = True
) -> logging.Logger:
    """
    Setup logging for the medical dataset generator.
    
    This is the main entry point for setting up logging throughout the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        structured_logging: Whether to use structured JSON logging
    
    Returns:
        Configured logger instance
    """
    medical_logger = MedicalDatasetLogger(
        log_level=log_level,
        log_file=log_file,
        console_output=console_output,
        structured_logging=structured_logging
    )
    
    return medical_logger.get_logger()


def get_logger(name: str = "medical_dataset_generator") -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically module name)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Create a default logger instance for use throughout the application
default_logger = get_logger()