# ML-Go Optuna

## Introduction

> [!IMPORTANT]
> This repo serves as the official repo for ML-Go (Exploitation based ML-Master).

Optuna 框架为了希望模型在探索新的未知策略可以在**充分探索并且思考已知策略的结果和思考**，为模型的训练代码提供了**超参数调优**和**日志记录分析**等核心功能，利用 Rule-based 的现有的机器学习的探索工具实现更加充分的 exploitation，为后续 Agent 的 Reflection, Refinement and Exploration 奠定数据基础。

## Modules

### Optuna Tuner

实现训练脚本输入，实现自动化参数优化并且得到最佳的超参数组合

### Optuna Logger

分析训练的日志和相关数据，进行数据分析计算更深层次的指标等等

### Optuna Introspector

根据 Tuner 得到的超参数优化的原始数据和 Logger 部分得到的其他指标，利用模型 (deepseek-r1) 生成详细的分析报告。

## Todo List

- [x] Finish Optuna Logger
- [x] Finish Optuna Introspector
- Optimize the prompt and strict the model's response format.
- Integrate this module into one **single exploitation-reflection pipeline**
    - Integrate that pipeline into MCTS


