class DataProcessingError(Exception):
    """Raised when there's an error in data processing"""
    pass

class ModelTrainingError(Exception):
    """Raised when there's an error in model training"""
    pass

class PredictionError(Exception):
    """Raised when there's an error in making predictions"""
    pass

class DatabaseError(Exception):
    """Raised when there's an error in database operations"""
    pass