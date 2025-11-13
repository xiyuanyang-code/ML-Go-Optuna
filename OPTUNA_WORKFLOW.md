# Optuna 优化器使用文档

## 概述

本项目提供了一个基于 Optuna 的超参数优化框架，包含以下核心模块：

1. **OptunaTuner** (`optuna-tuner/tuner.py`) - 超参数优化器
2. **OptunaLogger** (`optuna-tuner/logger.py`) - 优化结果分析器
3. **OptunaIntrospector** (`optuna-tuner/introspector.py`) - AI 驱动的优化结果分析器

## 固定 Workflow 流程

### 1. 超参数优化 (OptunaTuner)

使用 `OptunaTuner` 进行超参数优化：

```bash
python optuna-tuner/tuner.py <训练脚本路径> --trials <试验次数> --metric <优化指标>
```

参数说明：
- `<训练脚本路径>`: 需要优化的训练脚本
- `--trials`: 试验次数，默认为 50
- `--metric`: 优化的指标名称，默认为 accuracy

此步骤会在 `optuna_trials/` 目录下生成时间戳文件夹，包含：
- `best_params.json`: 最佳参数
- `optimization_summary.json`: 优化摘要
- `all_trials.json`: 所有试验结果
- `trial_*` 文件夹: 每个试验的详细结果

### 2. 结果分析 (OptunaLogger)

使用 `OptunaLogger` 对优化结果进行统计分析：

```bash
python optuna-tuner/logger.py <优化结果路径> --output <输出文件> --format <格式>
```

参数说明：
- `<优化结果路径>`: OptunaTuner 生成的时间戳文件夹路径
- `--output`: 输出文件路径，默认为 optimization_analysis_report.json
- `--format`: 输出格式 (json/txt)，默认为 json

分析内容包括：
- 参数重要性分析
- 性能趋势分析
- 参数相关性分析
- 统计报告生成

### 3. AI 深度分析 (OptunaIntrospector)

使用 `OptunaIntrospector` 通过 AI 模型对优化结果进行深度分析：

```bash
python optuna-tuner/introspector.py <优化结果路径> --output <输出文件>
```

参数说明：
- `<优化结果路径>`: OptunaTuner 生成的时间戳文件夹路径
- `--output`: 输出文件路径，默认为 optimization_analysis_report.md

此步骤需要配置环境变量：
- `INTROSPECTOR_API_KEY`: AI 模型 API 密钥
- `INTROSPECTOR_BASE_URL`: AI 模型 API 基础 URL

分析内容包括：
- 优化过程的关键发现
- 超参数重要性和关系的洞察
- 性能趋势和收敛性分析
- 未来优化运行的建议
- 超参数搜索空间优化建议
- 潜在问题或异常检测

## 使用示例

### 完整 Workflow 示例

1. 运行超参数优化：
```bash
python optuna-tuner/tuner.py test/pca_train.py --trials 20 --metric accuracy
```

2. 假设上一步生成了 `optuna_trials/20230515_143022` 文件夹，进行结果分析：
```bash
python optuna-tuner/logger.py optuna_trials/20230515_143022 --output analysis_report.json --format json
```

3. 进行 AI 深度分析：
```bash
export INTROSPECTOR_API_KEY=your_api_key
export INTROSPECTOR_BASE_URL=your_api_base_url
python optuna-tuner/introspector.py optuna_trials/20230515_143022 --output ai_analysis_report.md
```

## 模块详细说明

### OptunaTuner (tuner.py)

核心功能：
- 支持动态超参数搜索空间定义（通过 JSON 文件）
- 支持多种参数类型（int, float, categorical, bool）
- 自动实验日志记录和结果跟踪
- 基于时间的输出目录组织
- 可配置的优化策略和剪枝

配置文件格式：
```json
{
  "hyperparameter": {
    "参数名": {
      "type": "参数类型",
      "upperbound": "上界",
      "lowerbound": "下界",
      "space": "离散值空间",
      "default": "默认值",
      "description": "参数描述"
    }
  }
}
```

### OptunaLogger (logger.py)

核心功能：
- 超参数重要性分析
- 性能趋势可视化
- 参数与指标的相关性分析
- 优化历史跟踪
- 详细的统计报告生成

输出格式：
- JSON 格式：包含完整的统计分析数据
- TXT 格式：人类可读的文本报告

### OptunaIntrospector (introspector.py)

核心功能：
- 使用 AI 模型（如 GPT）分析超参数优化结果
- 生成包含洞察和建议的综合报告
- 支持不同输出格式
- 可配置的 AI API 集成

## 环境配置

1. 安装依赖：
```bash
pip install optuna openai python-dotenv numpy
```

2. 配置 AI 分析环境变量：
```bash
export INTROSPECTOR_API_KEY=your_api_key
export INTROSPECTOR_BASE_URL=your_api_base_url  # 可选，用于配置自定义 API 地址
```

## 注意事项

1. 训练脚本需要支持命令行参数输入超参数
2. 训练脚本需要在执行完成后输出包含指标的 `result.json` 文件
3. 进行 AI 分析时需要有效的 API 密钥
4. 优化结果文件夹路径需要与分析器参数匹配