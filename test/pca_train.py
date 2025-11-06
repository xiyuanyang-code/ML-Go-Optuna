#!/usr/bin/env python3
"""
PCA Hyperparameter Optimization Training Script

This script demonstrates PCA dimensionality reduction with hyperparameter
optimization for downstream classification performance. It's designed to work
with Optuna-based hyperparameter tuning framework.

Usage:
    python pca_train.py --n_components 10 --whiten True --solver auto --tol 0.001 --output_dir results
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def load_data(data_path: str = "data/mock_classification_data.csv") -> Tuple[np.ndarray, np.ndarray]:
    """Load training data from CSV file
    
    Args:
        data_path: Path to the CSV data file
        
    Returns:
        Tuple of (features, target) as numpy arrays
    """
    # Try different path resolutions
    possible_paths = [
        data_path,
        f"../{data_path}",
        f"../data/mock_classification_data.csv",
        "data/mock_classification_data.csv",
        os.path.join(os.path.dirname(__file__), "..", "data", "mock_classification_data.csv")
    ]
    
    actual_path = None
    for path in possible_paths:
        if os.path.exists(path):
            actual_path = path
            break
    
    if actual_path is None:
        print(f"Error: Data file not found. Tried paths:")
        for path in possible_paths:
            print(f"  - {path}")
        sys.exit(1)
    
    try:
        df = pd.read_csv(actual_path)
        X = df.drop('target', axis=1).values
        y = df['target'].values
        print(f"Loaded data from: {actual_path}")
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def create_model(n_components: int, whiten: bool, svd_solver: str, tol: float, 
                random_state: int = 42) -> Pipeline:
    """Create a PCA + RandomForest pipeline
    
    Args:
        n_components: Number of PCA components to retain
        whiten: Whether to apply whitening in PCA
        solver: PCA solver method ('auto', 'full', 'arpack', 'randomized')
        tol: Tolerance for PCA convergence
        random_state: Random seed for reproducibility
        
    Returns:
        Scikit-learn Pipeline with PCA and RandomForest
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(
            n_components=n_components,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            random_state=random_state
        )),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    return pipeline

def train_and_evaluate(model: Pipeline, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Train the model and evaluate performance
    
    Args:
        model: Scikit-learn pipeline to train
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        Dictionary containing evaluation metrics and model info
    """
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get PCA explained variance ratio
    pca = model.named_steps['pca']
    explained_variance_ratio = pca.explained_variance_ratio_.tolist()
    cumulative_variance = np.cumsum(explained_variance_ratio).tolist()
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return {
        'accuracy': float(accuracy),
        'training_time': float(training_time),
        'n_components': pca.n_components_,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'classification_report': report,
        'feature_importance': model.named_steps['classifier'].feature_importances_.tolist()
    }

def save_results(results: Dict[str, Any], output_dir: str):
    """Save training results to JSON file
    
    Args:
        results: Dictionary containing training results
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "result.json")
    
    # Save results to JSON
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {result_path}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Training time: {results['training_time']:.2f}s")
    print(f"Cumulative explained variance: {results['cumulative_variance'][-1]:.4f}")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description='PCA Hyperparameter Training')
    parser.add_argument('--n_components', type=int, required=True,
                        help='Number of PCA components to retain')
    parser.add_argument('--whiten', type=str, choices=['True', 'False'], required=True,
                        help='Whether to apply whitening in PCA')
    parser.add_argument('--svd_solver', type=str, choices=['auto', 'full', 'arpack', 'randomized'], 
                        required=True, help='PCA SVD solver method')
    parser.add_argument('--tol', type=float, required=True,
                        help='Tolerance for PCA convergence')
    parser.add_argument('--data_path', type=str, default="data/mock_classification_data.csv",
                        help='Path to training data CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_arguments()
    
    # Convert string boolean to actual boolean
    whiten = args.whiten == 'True'
    
    print(f"Starting PCA training with parameters:")
    print(f"  n_components: {args.n_components}")
    print(f"  whiten: {whiten}")
    print(f"  svd_solver: {args.svd_solver}")
    print(f"  tol: {args.tol}")
    print(f"  data_path: {args.data_path}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  seed: {args.seed}")
    
    # Load data
    X, y = load_data(args.data_path)
    print(f"Loaded data with shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Create model
    model = create_model(
        n_components=args.n_components,
        whiten=whiten,
        svd_solver=args.svd_solver,
        tol=args.tol,
        random_state=args.seed
    )
    
    # Train and evaluate
    results = train_and_evaluate(model, X_train, y_train, X_test, y_test)
    
    # Save results
    save_results(results, args.output_dir)

if __name__ == "__main__":
    main()