from sklearn.preprocessing import MinMaxScaler
import logging


def normalize(input_data):
    """Normalizes data"""

    # Initalize logger
    logger = logging.getLogger(__name__)

    # Convert 1D list to 2D list where every element is its own list
    values = list(map(lambda el:[el], input_data))
    logger.debug(f"Values:\nFirst 5:\n{values[:5]}\nLast 5:\n{values[-5:]}")

    # Initialize and fit scaler
    scaler = MinMaxScaler()
    scaler = scaler.fit(values)
    logger.debug(f"Min: {scaler.data_min_}, Max: {scaler.data_max_}")

    # Normalize data
    normalized = scaler.transform(values)
    logger.debug(f"Normalized Data:\nFirst 5:\n{normalized[:5]}\nLast 5:\n{normalized[-5:]}")

    # Flatten data and convert to python list
    normalized = normalized.flatten()
    output_data = normalized.tolist()
    
    return output_data
