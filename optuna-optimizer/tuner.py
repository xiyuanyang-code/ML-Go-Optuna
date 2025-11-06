"""
Optuna-based Hyperparameter Optimization Framework

This module provides a flexible hyperparameter optimization framework using Optuna.
It supports dynamic search space configuration via JSON files and can optimize
any training script that accepts hyperparameters via command-line arguments.

Key Features:
- Dynamic hyperparameter search space definition via JSON
- Support for multiple parameter types (int, float, categorical, bool)
- Automatic experiment logging and result tracking
- Time-based output directory organization
- Configurable optimization strategies and pruning
- Comprehensive result analysis and reporting

Author: ML-Go Optimization Team
Version: 1.0
"""

import optuna
import subprocess
import json
import os
import argparse
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OptunaTuner:
    """
    Advanced Optuna-based hyperparameter optimization framework.

    This class provides a comprehensive interface for optimizing machine learning
    models using Optuna. It supports dynamic hyperparameter search space definition,
    automatic experiment tracking, and flexible result management.

    Attributes:
        train_script (str): Path to the training script to optimize
        output_dir (str): Base directory for saving optimization results
        hyperparameter_config (Dict): Loaded hyperparameter configuration
        study_name (str): Name for the Optuna study
        timestamp (str): Timestamp for current optimization run

    Example:
        >>> tuner = OptunaTuner(
        ...     train_script="test/pca_train.py",
        ...     hyperparameter_file="test/pca_hyperparameters.json"
        ... )
        >>> study = tuner.run(n_trials=50)
    """

    def __init__(
        self,
        train_script: str = "model_train.py",
        hyperparameter_file: Optional[str] = None,
        output_dir: str = "optuna_trials",
        study_name: Optional[str] = None,
        metric_name: str = "accuracy",
    ):
        """
        Initialize the OptunaTuner.

        Args:
            train_script (str): Path to the training script that accepts hyperparameters
                via command-line arguments. Default is "model_train.py".
            hyperparameter_file (Optional[str]): Path to JSON file containing hyperparameter
                search space definition. If None, uses default CNN configuration.
            output_dir (str): Base directory for saving optimization results.
                Results will be saved in timestamped subdirectories. Default is "optuna_trials".
            study_name (Optional[str]): Name for the Optuna study. If None, generates
                a unique name based on the training script.
            metric_name (str): Name of the metric to optimize. Default is "accuracy".
        """

        self.train_script = train_script
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(output_dir, self.timestamp)
        os.makedirs(self.results_dir, exist_ok=True)

        # Load hyperparameter configuration
        self.hyperparameter_config = self._load_hyperparameter_config(
            hyperparameter_file
        )

        # Set study name
        if study_name is None:
            script_name = os.path.splitext(os.path.basename(train_script))[0]
            self.study_name = f"{script_name}_hyperopt_{self.timestamp}"
        else:
            self.study_name = study_name

        # 设置优化指标名称
        self.metric_name = metric_name

        logger.info(f"OptunaTuner initialized with: {train_script}")
        logger.info(f"Optimizing metric: {self.metric_name}")
        logger.info(f"Results will be saved to: {self.results_dir}")

    def _load_hyperparameter_config(
        self, hyperparameter_file: Optional[str]
    ) -> Dict[str, Any]:
        """
        Load hyperparameter configuration from JSON file or use default.

        Args:
            hyperparameter_file: Path to JSON configuration file

        Returns:
            Dictionary containing hyperparameter configuration
        """
        if hyperparameter_file and os.path.exists(hyperparameter_file):
            try:
                with open(hyperparameter_file, "r") as f:
                    config = json.load(f)
                logger.info(f"Loaded hyperparameter config from: {hyperparameter_file}")
                
                # 验证配置格式
                if not isinstance(config, dict):
                    logger.error(f"Invalid config format in {hyperparameter_file}: expected dict, got {type(config)}")
                    logger.info("Using default configuration")
                    return self._get_default_config()
                
                if "hyperparameter" not in config:
                    logger.warning(f"Missing 'hyperparameter' key in {hyperparameter_file}")
                    logger.info("Using default configuration")
                    return self._get_default_config()
                
                return config
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {hyperparameter_file}: {e}")
                logger.info("Using default configuration")
                return self._get_default_config()
            except Exception as e:
                logger.error(f"Failed to load config from {hyperparameter_file}: {e}")
                logger.info("Using default configuration")
                return self._get_default_config()

        logger.info("No hyperparameter file provided or file not found, using default configuration")
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default hyperparameter configuration.

        Returns:
            Dictionary containing default hyperparameter configuration
        """
        # Default configuration that is more generic
        return {
            "hyperparameter": {
                "learning_rate": {
                    "type": "float",
                    "upperbound": 0.1,
                    "lowerbound": 0.0001,
                    "log": True,
                    "default": 0.01,
                    "description": "Learning rate for the optimizer"
                },
                "batch_size": {
                    "type": "categorical",
                    "space": [16, 32, 64, 128],
                    "default": 32,
                    "description": "Batch size for training"
                },
                "epochs": {
                    "type": "int",
                    "upperbound": 50,
                    "lowerbound": 5,
                    "default": 10,
                    "description": "Number of training epochs"
                },
            },
            "metadata": {
                "description": "Default hyperparameter configuration",
                "version": "1.0"
            }
        }

    def _suggest_hyperparameters(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters based on configuration.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        hyperparams = {}
        config = self.hyperparameter_config.get("hyperparameter", {})

        for param_name, param_config in config.items():
            param_type = param_config.get("type")
            
            # 添加参数验证
            if not param_type:
                logger.warning(f"No type specified for parameter: {param_name}, skipping")
                continue

            try:
                if param_type == "int":
                    lowerbound = param_config.get("lowerbound")
                    upperbound = param_config.get("upperbound")
                    
                    # 验证边界值
                    if lowerbound is None or upperbound is None:
                        logger.error(f"Missing bounds for int parameter: {param_name}")
                        if "default" in param_config:
                            hyperparams[param_name] = param_config["default"]
                        continue
                    
                    hyperparams[param_name] = trial.suggest_int(
                        param_name,
                        lowerbound,
                        upperbound,
                        step=param_config.get("step", 1),
                    )
                elif param_type == "float":
                    lowerbound = param_config.get("lowerbound")
                    upperbound = param_config.get("upperbound")
                    
                    # 验证边界值
                    if lowerbound is None or upperbound is None:
                        logger.error(f"Missing bounds for float parameter: {param_name}")
                        if "default" in param_config:
                            hyperparams[param_name] = param_config["default"]
                        continue
                    
                    if param_config.get("log", False):
                        hyperparams[param_name] = trial.suggest_float(
                            param_name,
                            lowerbound,
                            upperbound,
                            log=True,
                        )
                    else:
                        hyperparams[param_name] = trial.suggest_float(
                            param_name,
                            lowerbound,
                            upperbound,
                        )
                elif param_type == "categorical" or param_type == "str":
                    space = param_config.get("space", param_config.get("choices"))
                    
                    # 验证空间值
                    if not space:
                        logger.error(f"Missing space for categorical parameter: {param_name}")
                        if "default" in param_config:
                            hyperparams[param_name] = param_config["default"]
                        continue
                    
                    hyperparams[param_name] = trial.suggest_categorical(
                        param_name,
                        space,
                    )
                elif param_type == "bool":
                    space = param_config.get("space", [True, False])
                    hyperparams[param_name] = trial.suggest_categorical(
                        param_name, space
                    )
                else:
                    logger.warning(
                        f"Unknown parameter type: {param_type} for {param_name}"
                    )
                    # 使用默认值
                    if "default" in param_config:
                        hyperparams[param_name] = param_config["default"]

            except Exception as e:
                logger.error(f"Error suggesting parameter {param_name}: {e}")
                # Use default value if suggestion fails
                if "default" in param_config:
                    hyperparams[param_name] = param_config["default"]
                else:
                    logger.warning(f"No default value for parameter: {param_name}, skipping")

        return hyperparams

    def _build_command(
        self, hyperparams: Dict[str, Any], trial_number: int
    ) -> List[str]:
        """
        Build command line arguments for training script.

        Args:
            hyperparams: Dictionary of hyperparameters
            trial_number: Current trial number

        Returns:
            List of command line arguments
        """
        trial_dir = os.path.join(self.results_dir, f"trial_{trial_number}")
        os.makedirs(trial_dir, exist_ok=True)

        cmd = ["python", self.train_script]

        # Add hyperparameters to command
        for param_name, param_value in hyperparams.items():
            cmd.extend([f"--{param_name}", str(param_value)])

        # Add output directory and seed
        cmd.extend(["--output_dir", trial_dir, "--seed", "42"])

        return cmd

    def _execute_training(
        self, cmd: List[str], trial_number: int
    ) -> Optional[Dict[str, Any]]:
        """
        Execute training script and capture results.

        Args:
            cmd: Command line arguments to execute
            trial_number: Current trial number

        Returns:
            Dictionary containing training results or None if failed
        """
        logger.info(f"[Trial {trial_number}] Running: {' '.join(cmd)}")

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600
            )  # 1 hour timeout
            execution_time = time.time() - start_time

            # Log stdout and stderr for debugging
            if result.stdout:
                logger.debug(f"[Trial {trial_number}] STDOUT: {result.stdout[:500]}...")
            if result.stderr:
                logger.warning(
                    f"[Trial {trial_number}] STDERR: {result.stderr[:500]}..."
                )

            # Check if training succeeded
            if result.returncode != 0:
                logger.error(
                    f"[Trial {trial_number}] Training failed with return code: {result.returncode}"
                )
                logger.error(f"[Trial {trial_number}] STDERR: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error(f"[Trial {trial_number}] Training timed out")
            return None
        except Exception as e:
            logger.error(f"[Trial {trial_number}] Error during training: {e}")
            return None

        # Read results
        trial_dir = os.path.join(self.results_dir, f"trial_{trial_number}")
        result_path = os.path.join(trial_dir, "result.json")

        if not os.path.exists(result_path):
            logger.error(f"[Trial {trial_number}] Result file not found: {result_path}")
            # 尝试其他可能的结果文件名
            possible_result_files = ["result.json", "results.json", "metrics.json", "output.json"]
            for file_name in possible_result_files:
                alt_result_path = os.path.join(trial_dir, file_name)
                if os.path.exists(alt_result_path):
                    result_path = alt_result_path
                    logger.info(f"[Trial {trial_number}] Using alternative result file: {result_path}")
                    break
            else:
                logger.error(f"[Trial {trial_number}] No result file found in {trial_dir}")
                return None

        try:
            with open(result_path) as f:
                data = json.load(f)
            
            # 确保数据是字典格式
            if not isinstance(data, dict):
                logger.error(f"[Trial {trial_number}] Result data is not a dictionary: {type(data)}")
                return None
                
            data["execution_time"] = execution_time
            return data
        except json.JSONDecodeError as e:
            logger.error(f"[Trial {trial_number}] Invalid JSON in result file: {e}")
            return None
        except Exception as e:
            logger.error(f"[Trial {trial_number}] Error reading results: {e}")
            return None

    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        Objective function for Optuna optimization.

        This method is called by Optuna for each trial. It suggests hyperparameters,
        executes the training script, and returns the optimization metric.

        Args:
            trial: Optuna trial object

        Returns:
            The optimization metric value (higher is better by default)
        """
        # Suggest hyperparameters based on configuration
        hyperparams = self._suggest_hyperparameters(trial)

        # Build and execute training command
        cmd = self._build_command(hyperparams, trial.number)
        results = self._execute_training(cmd, trial.number)

        if results is None:
            # Return a very low value if training failed
            return float('-inf') if self.direction == "maximize" else float('inf')

        # Extract optimization metric
        # First try the specified metric name, then fall back to common metrics
        metric_value = None
        
        # 优先使用指定的指标名称
        if self.metric_name in results:
            metric_value = float(results[self.metric_name])
            logger.info(f"Using specified metric '{self.metric_name}': {metric_value}")
        else:
            # Look for common metrics as fallback
            for metric_name in [
                "accuracy",
                "f1_score",
                "roc_auc",
                "val_accuracy",
                "test_accuracy",
                "precision",
                "recall",
                "mse",
                "mae",
                "r2_score",
            ]:
                if metric_name in results:
                    metric_value = float(results[metric_name])
                    logger.info(f"Using fallback metric '{metric_name}': {metric_value}")
                    break
        
        # 如果没有找到任何指标，记录错误并返回极值
        if metric_value is None:
            logger.error(f"Metric '{self.metric_name}' not found in results and no fallback available")
            return float('-inf') if self.direction == "maximize" else float('inf')

        # Store additional information for analysis
        trial.set_user_attr(
            "trial_dir", os.path.join(self.results_dir, f"trial_{trial.number}")
        )
        trial.set_user_attr("execution_time", results.get("execution_time", 0.0))
        trial.set_user_attr("hyperparameters", hyperparams)

        # Store all available metrics for later analysis
        for key, value in results.items():
            if isinstance(value, (int, float)):
                trial.set_user_attr(key, value)

        return metric_value

    def run(
        self,
        n_trials: int = 50,
        storage: Optional[str] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        direction: str = "maximize",
    ) -> optuna.Study:
        """
        Run hyperparameter optimization.

        Args:
            n_trials (int): Number of optimization trials to run. Default is 50.
            storage (Optional[str]): Database URL for storing study results.
                If None, uses in-memory storage. Default is None.
            sampler (Optional[optuna.samplers.BaseSampler]): Sampling strategy.
                If None, uses TPESampler. Default is None.
            pruner (Optional[optuna.pruners.BasePruner]): Pruning strategy.
                If None, uses MedianPruner. Default is None.
            direction (str): Optimization direction ("maximize" or "minimize").
                Default is "maximize".

        Returns:
            optuna.Study: Completed Optuna study object with results

        Raises:
            ValueError: If direction is not "maximize" or "minimize"
        """
        if direction not in ["maximize", "minimize"]:
            raise ValueError("Direction must be 'maximize' or 'minimize'")

        # Set default sampler and pruner
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=42)
        if pruner is None:
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)

        # Create study
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            study_name=self.study_name,
            storage=storage,
            load_if_exists=True,
        )

        # 保存direction属性以便在objective中使用
        self.direction = direction

        logger.info(f"Starting {direction} optimization with {n_trials} trials")
        logger.info(f"Study name: {self.study_name}")
        logger.info(f"Optimizing metric: {self.metric_name}")
        if storage:
            logger.info(f"Storage: {storage}")

        # Run optimization
        start_time = time.time()
        study.optimize(self.objective, n_trials=n_trials)
        optimization_time = time.time() - start_time

        # Log results
        logger.info(f"Optimization completed in {optimization_time:.2f}s")
        logger.info(f"Best {direction} value: {study.best_value:.4f}")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best parameters: {study.best_params}")

        # Save comprehensive results
        self._save_results(study, optimization_time)

        return study

    def _save_results(self, study: optuna.Study, optimization_time: float):
        """
        Save optimization results to files.

        Args:
            study: Completed Optuna study
            optimization_time: Total optimization time in seconds
        """
        # Save best parameters
        best_params_path = os.path.join(self.results_dir, "best_params.json")
        with open(best_params_path, "w") as f:
            json.dump(study.best_params, f, indent=2)

        # Save optimization summary
        summary = {
            "study_name": self.study_name,
            "best_value": study.best_value,
            "best_trial": study.best_trial.number,
            "best_parameters": study.best_params,
            "best_trial_dir": study.best_trial.user_attrs.get("trial_dir"),
            "n_trials": len(study.trials),
            "optimization_time": optimization_time,
            "direction": study.direction,
            "hyperparameter_config": self.hyperparameter_config,
            "timestamp": self.timestamp,
            "train_script": self.train_script,
        }

        summary_path = os.path.join(self.results_dir, "optimization_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save all trial results
        trials_data = []
        for trial in study.trials:
            trial_info = {
                "trial_number": trial.number,
                "value": trial.value,
                "parameters": trial.params,
                "state": trial.state.name,
                "datetime_start": (
                    trial.datetime_start.isoformat() if trial.datetime_start else None
                ),
                "datetime_complete": (
                    trial.datetime_complete.isoformat()
                    if trial.datetime_complete
                    else None
                ),
                "user_attrs": trial.user_attrs,
            }
            trials_data.append(trial_info)

        trials_path = os.path.join(self.results_dir, "all_trials.json")
        with open(trials_path, "w") as f:
            json.dump(trials_data, f, indent=2)

        logger.info(f"Results saved to: {self.results_dir}")
        logger.info(f"  - Best parameters: {best_params_path}")
        logger.info(f"  - Optimization summary: {summary_path}")
        logger.info(f"  - All trials: {trials_path}")


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Optuna-based hyperparameter optimization framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Only required argument is the training script (optional with auto-detection)
    parser.add_argument(
        "train_script",
        type=str,
        help="Path to the training script to optimize (optional, will search in common locations)",
    )

    # Optional arguments with sensible defaults
    parser.add_argument(
        "--trials", type=int, default=50, help="Number of optimization trials"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Database URL for storing study results",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["maximize", "minimize"],
        default="maximize",
        help="Optimization direction",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="tpe",
        choices=["tpe", "random", "cmaes"],
        help="Sampling strategy",
    )
    parser.add_argument(
        "--pruner",
        type=str,
        default="median",
        choices=["median", "hyperband", "threshold", "none"],
        help="Pruning strategy",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--metric", type=str, default="accuracy", help="Metric name to optimize (default: accuracy)"
    )

    args = parser.parse_args()

    # Auto-detect hyperparameter file
    script_dir = os.path.dirname(args.train_script)
    script_name = os.path.splitext(os.path.basename(args.train_script))[0]

    hyperparameter_file = None
    # Try different naming patterns for hyperparameter files, in order of preference
    hyperparameter_patterns = [
        f"{script_dir}/{script_name}_hyperparameters.json",  # Script-specific config
        f"{script_dir}/hyperparameters.json",                # General config
        f"{script_dir}/params.json",                         # Alternative config
    ]
    for pattern in hyperparameter_patterns:
        if os.path.exists(pattern):
            hyperparameter_file = pattern
            print(f"Auto-detected hyperparameter config: {pattern}")
            break

    # Create sampler based on choice
    sampler_map = {
        "tpe": optuna.samplers.TPESampler(seed=args.seed),
        "random": optuna.samplers.RandomSampler(seed=args.seed),
        "cmaes": optuna.samplers.CmaEsSampler(seed=args.seed),
    }

    # Create pruner based on choice
    pruner_map = {
        "median": optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        "hyperband": optuna.pruners.HyperbandPruner(),
        "threshold": optuna.pruners.ThresholdPruner(lower=0.0),
        "none": optuna.pruners.NopPruner(),
    }

    # Initialize tuner with timestamp-based output directory
    tuner = OptunaTuner(
        train_script=args.train_script,
        hyperparameter_file=hyperparameter_file,
        metric_name=args.metric,  # 传递指标名称
    )

    print(f"Starting optimization with {args.trials} trials...")
    print(f"Training script: {args.train_script}")
    print(f"Optimizing metric: {args.metric}")
    if hyperparameter_file:
        print(f"Hyperparameter config: {hyperparameter_file}")
    print(f"Results will be saved to: {tuner.results_dir}")

    # Run optimization
    study = tuner.run(
        n_trials=args.trials,
        storage=args.storage,
        sampler=sampler_map[args.sampler],
        pruner=pruner_map[args.pruner],
        direction=args.direction,
    )

    return study


if __name__ == "__main__":
    study = main()
