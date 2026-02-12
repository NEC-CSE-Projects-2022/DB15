import pickle
import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_scaler():
    try:
        # Load scaler
        with open('scaler1.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        # Test data
        test_data = np.array([[30, 7.0, 7, 50, 5, 78, 7500, 128, 84]])
        
        # Try transformation
        scaled_data = scaler.transform(test_data)
        
        logger.info("Scaler test successful!")
        logger.info(f"Original data shape: {test_data.shape}")
        logger.info(f"Scaled data shape: {scaled_data.shape}")
        return True
        
    except Exception as e:
        logger.error(f"Error testing scaler: {str(e)}")
        return False

if __name__ == "__main__":
    test_scaler()