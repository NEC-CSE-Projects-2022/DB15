from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_and_save_scaler():
    try:
        # Create example data matching your features
        # Using realistic ranges for each feature
        example_data = np.array([
            # Age, Sleep Duration, Quality, Activity, Stress, Heart Rate, Steps, Systolic, Diastolic
            [25, 7.5, 7, 60, 4, 75, 8000, 120, 80],
            [35, 6.5, 6, 45, 6, 85, 6000, 130, 85],
            [45, 8.0, 8, 30, 3, 70, 10000, 125, 82],
            [55, 5.5, 5, 20, 7, 80, 4000, 140, 90]
        ])

        # Create and fit scaler
        scaler = StandardScaler()
        scaler.fit(example_data)
        
        # Save scaler
        with open('scaler1.pkl', 'wb') as f:
            pickle.dump(scaler, f)
            
        logger.info("Scaler created and saved successfully!")
        
        # Test the scaler
        test_data = np.array([[30, 7.0, 7, 50, 5, 78, 7500, 128, 84]])
        scaled_data = scaler.transform(test_data)
        logger.info("Scaler test successful!")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating scaler: {str(e)}")
        return False

if __name__ == "__main__":
    create_and_save_scaler()