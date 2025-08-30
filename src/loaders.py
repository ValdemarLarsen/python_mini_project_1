# Enhanced Data Loaders with Error Handling and Additional Features
import pandas as pd
import json
import os
from pathlib import Path
import logging
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load a CSV file into a Pandas DataFrame with enhanced error handling
    
    Args:
        file_path (str): Path to the CSV file
        **kwargs: Additional arguments to pass to pd.read_csv()
    
    Returns:
        pd.DataFrame: Loaded data
    
    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.EmptyDataError: If file is empty
        Exception: For other loading errors
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        # Set default parameters for robust CSV reading
        default_params = {
            'encoding': 'utf-8',
            'sep': ',',
            'skipinitialspace': True,
            'na_values': ['', 'NA', 'N/A', 'null', 'NULL', 'None']
        }
        
        # Merge with user-provided parameters
        params = {**default_params, **kwargs}
        
        logger.info(f"Loading CSV file: {file_path}")
        df = pd.read_csv(file_path, **params)
        
        # Validate the loaded data
        if df.empty:
            logger.warning(f"Loaded CSV file is empty: {file_path}")
        else:
            logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
        
        return df
        
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {file_path}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading CSV {file_path}: {e}")
        raise

def load_excel(file_path: str, sheet_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Load an Excel file into a Pandas DataFrame with enhanced error handling
    
    Args:
        file_path (str): Path to the Excel file
        sheet_name (str, optional): Name of the sheet to load
        **kwargs: Additional arguments to pass to pd.read_excel()
    
    Returns:
        pd.DataFrame: Loaded data
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If specified sheet doesn't exist
        Exception: For other loading errors
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Excel file not found: {file_path}")
        
        # Set default parameters
        default_params = {
            'na_values': ['', 'NA', 'N/A', 'null', 'NULL', 'None'],
            'keep_default_na': True
        }
        
        if sheet_name:
            default_params['sheet_name'] = sheet_name
        
        # Merge with user-provided parameters
        params = {**default_params, **kwargs}
        
        logger.info(f"Loading Excel file: {file_path}")
        if sheet_name:
            logger.info(f"Loading sheet: {sheet_name}")
        
        df = pd.read_excel(file_path, **params)
        
        # Validate the loaded data
        if df.empty:
            logger.warning(f"Loaded Excel file is empty: {file_path}")
        else:
            logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
        
        return df
        
    except ValueError as e:
        if "sheet" in str(e).lower():
            logger.error(f"Sheet not found in Excel file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading Excel file {file_path}: {e}")
        raise

def load_json(file_path: str, normalize: bool = True, **kwargs) -> pd.DataFrame:
    """
    Load a JSON file into a Pandas DataFrame with enhanced error handling
    
    Args:
        file_path (str): Path to the JSON file
        normalize (bool): Whether to normalize nested JSON structures
        **kwargs: Additional arguments for JSON loading or normalization
    
    Returns:
        pd.DataFrame: Loaded data
    
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
        Exception: For other loading errors
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        logger.info(f"Loading JSON file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        if normalize and isinstance(data, (list, dict)):
            # Use json_normalize for nested structures
            df = pd.json_normalize(data, **kwargs)
        else:
            # Direct conversion
            df = pd.DataFrame(data)
        
        # Validate the loaded data
        if df.empty:
            logger.warning(f"Loaded JSON file resulted in empty DataFrame: {file_path}")
        else:
            logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
        
        return df
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        raise

def load_data_with_fallback(file_path: str) -> pd.DataFrame:
    """
    Load data from a file, automatically detecting format and applying fallbacks
    
    Args:
        file_path (str): Path to the data file
    
    Returns:
        pd.DataFrame: Loaded data
    """
    file_extension = Path(file_path).suffix.lower()
    
    try:
        if file_extension == '.csv':
            return load_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            return load_excel(file_path)
        elif file_extension == '.json':
            return load_json(file_path)
        else:
            # Try CSV as fallback
            logger.warning(f"Unknown file extension {file_extension}, trying CSV format")
            return load_csv(file_path)
            
    except Exception as e:
        logger.error(f"Failed to load file {file_path}: {e}")
        raise

def validate_data_format(df: pd.DataFrame, expected_columns: Optional[list] = None) -> Dict[str, Any]:
    """
    Validate the format and quality of loaded data
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        expected_columns (list, optional): List of expected column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'stats': {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append('DataFrame is empty')
    
    # Check for expected columns
    if expected_columns:
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f'Missing expected columns: {missing_columns}')
    
    # Check for high percentage of missing values
    if len(df) > 0:
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_percentage > 50:
            validation_results['issues'].append(f'High percentage of missing values: {missing_percentage:.1f}%')
    
    # Check for duplicate rows
    if df.duplicated().sum() > 0:
        validation_results['issues'].append(f'Found {df.duplicated().sum()} duplicate rows')
    
    return validation_results

# Convenience function for loading multiple files
def load_multiple_files(file_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Load multiple data files at once
    
    Args:
        file_paths (dict): Dictionary with name -> file_path mappings
    
    Returns:
        dict: Dictionary with name -> DataFrame mappings
    """
    datasets = {}
    
    for name, path in file_paths.items():
        try:
            logger.info(f"Loading {name} from {path}")
            datasets[name] = load_data_with_fallback(path)
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
            datasets[name] = None
    
    return datasets