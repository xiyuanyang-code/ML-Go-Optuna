# Optuna 超参数优化框架使用指南

## 概述

Optuna 优化器是一个全面的超参数优化框架，使用 Optuna 的高级采样和剪枝策略自动搜索最优的机器学习模型参数。它被设计为可与任何通过命令行参数接受超参数的训练脚本配合使用。

### 核心功能

Tuner 模块利用 Optuna 的机器学习调参库实现了在机器学习过程中 **参数搜索优化** 和 **结构优化** 两种 improvement 的分离，一方面我们希望搜索树保持良好的分散结构实现更多网络结构的有效搜索，另一方面我们希望 ML-Go 能够针对已经有的结构做充分的思考和分析，而这其中的第一步就是**充分的探索现在已有的网络结构**。

该框架具有以下优势：
- 后续提示词工程的优化和 Agent Self Reflection 的过程奠定数据基础
- 在结果上也可以帮助 Agent 最大化的利用当前模型的表现


## Configuration

为了适应 Optuna 调参框架，在训练之前需要首先做如下配置：

- 模型的训练脚本，支持 argparse 的超参数输入，输出对应指标的 evaluation 和 submission 的文件
- 相同目录下超参数和评测指标的 json 文件

### 超参数 JSON 文件配置

```json
{
  "hyperparameter": {
    "parameter_name": {
      "type": "int|float|categorical|bool",
      "upperbound": 100,
      "lowerbound": 1,
      "step": 1,
      "default": 10,
      "space": ["option1", "option2"],
      "log": true,
      "description": "参数描述"
    }
  },
  "metadata": {
    "experiment_name": "my_experiment",
    "description": "实验描述",
    "objective": "maximize_accuracy",
    "dataset": "my_dataset.csv",
    "model_type": "my_model_type",
    "version": "1.0"
  }
}
```

#### 参数类型

- **int**: 整数值，具有上下界和可选步长
- **float**: 浮点数值，可选择对数刻度
- **categorical**: 空间数组中的离散选择
- **bool**: 布尔值 (true/false)

#### 示例：PCA 配置

```json
{
  "hyperparameter": {
    "n_components": {
      "type": "int",
      "upperbound": 50,
      "lowerbound": 2,
      "step": 1,
      "default": 10,
      "description": "要保留的主成分数量"
    },
    "whiten": {
      "type": "bool",
      "space": [true, false],
      "default": false,
      "description": "是否应用白化"
    },
    "svd_solver": {
      "type": "categorical",
      "space": ["auto", "full", "arpack", "randomized"],
      "default": "auto",
      "description": "PCA SVD 求解器方法"
    },
    "tol": {
      "type": "float",
      "upperbound": 0.01,
      "lowerbound": 0.0001,
      "default": 0.001,
      "description": "收敛容差"
    }
  }
}
```

### 训练脚本要求

为了保证该框架的鲁棒性，模型的训练脚本可以**自由调整数据输入、模型训练框架脚本**等模块，只限制如下的接口：

- 模型必须利用 argparse 接受对应的超参数，注意超参数的设置需要和 json 格式中的设置保持一致。
  - 所有训练脚本都应接受：
    - `--output_dir`: 保存结果的目录
    - `--seed`: 用于重现性的随机种子
- 模型需要输出在 output_dir 输出两个文件：
    - `submission.csv`：方便后续直接提交结果到最终的提交平台
    - `result.json`：计算 evaluation 的指标，方便后续 optuna 进行超参数优化

```python
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
```

## 命令行参数

```bash
python optuna-optimizer/tuner.py [training_script] [options]

# 位置参数
training_script           # 训练脚本路径（可选，自动检测）

# 选项
--trials INT            # 优化试验次数（默认：50）
--direction STR          # 优化方向：maximize/minimize（默认：maximize）
--sampler STR           # 采样策略：tpe/random/cmaes（默认：tpe）
--pruner STR            # 剪枝策略：median/hyperband/threshold/none（默认：median）
--storage STR           # 持久化存储的数据库URL（默认：内存存储）
--seed INT              # 重现性的随机种子（默认：42）
--metric STR            # 要优化的指标名称（默认：accuracy）
```

> [!IMPORTANT]
> Optuna 模块讲嵌入自动化的评测流程中，因此后续的一些参数设置在实际使用中会选择**默认设置**，参数的选择为了提高后续代码的鲁棒性和可扩展性。

- 采样策略
  - **tpe**: 树形Parzen估计器（推荐）
  - **random**: 随机采样
  - **cmaes**: 基于CMA-ES的采样
- 剪枝策略
  - **median**: 基于中位数的剪枝（推荐）
  - **hyperband**: Hyperband剪枝
  - **threshold**: 基于阈值的剪枝
  - **none**: 不剪枝

```bash
### 自定义研究名称
python optuna-optimizer/tuner.py --study_name "my_experiment_v1"
```

```bash
### 最小化问题
python optuna-optimizer/tuner.py --direction minimize
```

```bash
### 指定优化指标
python optuna-optimizer/tuner.py --metric f1_score
```

## 结果分析

### 最佳参数

优化后，`best_params.json` 包含：

```json
{
  "learning_rate": 0.001,
  "batch_size": 64,
  "epochs": 20,
  "regularization": 0.1
}
```

### 优化摘要

`optimization_summary.json` 提供完整元数据：

```json
{
  "study_name": "pca_train_hyperopt_20251106_011609",
  "best_value": 0.705,
  "best_trial": 18,
  "best_parameters": {...},
  "n_trials": 50,
  "optimization_time": 35.96,
  "direction": "maximize",
  "timestamp": "20251106_011609",
  "train_script": "test/pca_train.py"
}
```

### 所有试验数据

`all_trials.json` 包含每个试验：

```json
[
  {
    "trial_number": 0,
    "value": 0.595,
    "parameters": {"n_components": 20, "whiten": true},
    "state": "COMPLETE",
    "datetime_start": "2025-11-06T01:16:09.715000",
    "datetime_complete": "2025-11-06T01:16:10.424000",
    "user_attrs": {
      "execution_time": 0.709,
      "trial_dir": "optuna_results_20251106_011609/trial_0"
    }
  }
]
```
