import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import os

def generate_mock_data(n_samples=1000, n_features=50, n_classes=3, random_state=42):
    """Generate mock dataset for PCA demonstration
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        n_classes: Number of classes for classification
        random_state: Random seed for reproducibility
    """
    np.random.seed(random_state)
    
    # Generate synthetic classification dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(10, n_features // 5),
        n_redundant=max(5, n_features // 10),
        n_classes=n_classes,
        random_state=random_state
    )
    
    # Create feature names
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Save to CSV
    output_path = os.path.join('data', 'mock_classification_data.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Mock data saved to {output_path}")
    print(f"Data shape: {df.shape}")
    print(f"Class distribution: {df['target'].value_counts().to_dict()}")
    
    return df

if __name__ == "__main__":
    generate_mock_data()