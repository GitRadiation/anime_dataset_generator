"""Custom exceptions for the project."""

class DataValidationError(Exception):
    """Raised when data validation fails."""

class GeneticAlgorithmError(Exception):
    """Base exception for genetic algorithm errors."""

class PopulationInitializationError(GeneticAlgorithmError):
    """Raised when population initialization fails."""

class EvolutionError(GeneticAlgorithmError):
    """Raised when evolution process fails."""

class DatabaseError(Exception):
    """Raised when a database operation fails."""