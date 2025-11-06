#!/usr/bin/env python3
import argparse
import json
import os
import time
import random
import pandas as pd
import numpy as np

def train_model(args):
    """Simple training function for testing the tuner."""
    print(f"Training with parameters: {vars(args)}")
    
    # 模拟训练过程
    time.sleep(1)
    
    # 生成一些随机结果
    # 使用参数来影响结果，使优化有意义
    learning_rate = getattr(args, 'learning_rate', 0.01)
    batch_size = getattr(args, 'batch_size', 32)
    epochs = getattr(args, 'epochs', 10)
    
    # 创建一个基于参数的"性能"指标
    # 这样不同的参数组合会产生不同的结果
    accuracy = 0.7 + (learning_rate * 10) + (batch_size / 1000) + (epochs / 100) + random.uniform(-0.1, 0.1)
    accuracy = max(0.0, min(1.0, accuracy))  # 限制在0-1之间
    
    f1_score = accuracy * 0.9 + random.uniform(-0.05, 0.05)
    f1_score = max(0.0, min(1.0, f1_score))
    
    # 生成模拟的submission.csv文件
    # 在实际应用中，这将是模型预测的结果
    num_samples = 1000
    submission_data = {
        'id': range(1, num_samples + 1),
        'prediction': np.random.choice([0, 1], size=num_samples, p=[1-accuracy, accuracy])
    }
    submission_df = pd.DataFrame(submission_data)
    
    # 保存submission.csv文件
    os.makedirs(args.output_dir, exist_ok=True)
    submission_df.to_csv(os.path.join(args.output_dir, "submission.csv"), index=False)
    
    # 保存指标到JSON文件
    results = {
        "accuracy": accuracy,
        "f1_score": f1_score,
        "training_time": time.time() - time.time() + 1.0,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }
    
    with open(os.path.join(args.output_dir, "result.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Training completed with accuracy: {accuracy:.4f}")
    print(f"Submission file saved to: {os.path.join(args.output_dir, 'submission.csv')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    train_model(args)