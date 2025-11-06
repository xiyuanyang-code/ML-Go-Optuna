"""
Optuna Logger Module

This module provides comprehensive analysis capabilities for Optuna optimization results.
It analyzes hyperparameter relationships, performance trends, and generates visualizations
to help understand the optimization process and results.

Key Features:
- Hyperparameter importance analysis
- Performance trend visualization
- Correlation analysis between parameters and metrics
- Optimization history tracking
- Detailed statistical reports

Author: ML-Go Optimization Team
Version: 1.0
"""

import json
import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import optuna

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OptunaLogger:
    """
    Comprehensive analysis tool for Optuna optimization results.

    This class provides methods to analyze hyperparameter optimization results,
    including parameter importance, performance trends, and correlation analysis.
    It generates detailed reports and can export results in various formats.

    Attributes:
        study_path (str): Path to the optimization study results
        study (optuna.Study): Loaded Optuna study object
        trials_data (List[Dict]): Processed trial data
        hyperparameter_config (Dict): Hyperparameter configuration
    """

    def __init__(self, study_path: str):
        """
        Initialize the OptunaLogger.

        Args:
            study_path (str): Path to the optimization study directory containing
                'all_trials.json' and 'optimization_summary.json' files.
        """
        self.study_path = study_path
        self.study = None
        self.trials_data = []
        self.hyperparameter_config = {}
        
        # Load study data
        self._load_study_data()
        
        logger.info(f"OptunaLogger initialized with study path: {study_path}")

    def _load_study_data(self):
        """Load study data from JSON files."""
        # Load all trials data
        trials_file = os.path.join(self.study_path, "all_trials.json")
        if os.path.exists(trials_file):
            with open(trials_file, "r") as f:
                self.trials_data = json.load(f)
            logger.info(f"Loaded {len(self.trials_data)} trials from {trials_file}")
        else:
            logger.error(f"Trials file not found: {trials_file}")
            raise FileNotFoundError(f"Trials file not found: {trials_file}")
            
        # Load hyperparameter configuration
        summary_file = os.path.join(self.study_path, "optimization_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                summary_data = json.load(f)
                self.hyperparameter_config = summary_data.get("hyperparameter_config", {})
            logger.info(f"Loaded hyperparameter configuration from {summary_file}")
        else:
            logger.warning(f"Summary file not found: {summary_file}")

    def analyze_parameter_importance(self) -> Dict[str, float]:
        """
        Analyze the importance of each hyperparameter in affecting the optimization metric.

        Returns:
            Dictionary mapping parameter names to their importance scores (higher is more important).
        """
        if not self.trials_data:
            logger.warning("No trial data available for parameter importance analysis")
            return {}
            
        # Extract parameter values and metric values
        param_values = {}
        metric_values = []
        
        # Get metric name from the first trial with a value
        metric_name = None
        valid_trials = [t for t in self.trials_data if t.get("value") is not None]
        if valid_trials:
            # Try to get metric name from user attributes
            first_trial = valid_trials[0]
            user_attrs = first_trial.get("user_attrs", {})
            # Look for common metric names in user attributes
            for key in user_attrs.keys():
                if key in ["accuracy", "f1_score", "precision", "recall", "loss", "mse", "mae"]:
                    metric_name = key
                    break
            # If not found, use "value" as the metric name
            if metric_name is None:
                metric_name = "value"
        
        for trial in valid_trials:
            params = trial.get("parameters", {})
            value = trial.get("value")
            
            if value is None:
                continue
                
            metric_values.append(value)
            
            for param_name, param_value in params.items():
                if param_name not in param_values:
                    param_values[param_name] = []
                param_values[param_name].append(param_value)
        
        if not param_values or not metric_values:
            logger.warning("No valid parameter or metric data for importance analysis")
            return {}
            
        # Calculate importance scores using correlation coefficients
        importance_scores = {}
        metric_array = np.array(metric_values)
        
        for param_name, values in param_values.items():
            param_array = np.array(values)
            
            # For categorical parameters, we need to convert to numerical values
            if param_array.dtype == object:
                # Convert categorical to numerical using label encoding
                unique_vals = list(np.unique(param_array))
                param_numerical = np.array([unique_vals.index(v) for v in param_array])
            else:
                param_numerical = param_array
            
            # Calculate correlation coefficient
            if len(np.unique(param_numerical)) > 1 and len(np.unique(metric_array)) > 1:
                correlation = np.corrcoef(param_numerical, metric_array)[0, 1]
                importance_scores[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                importance_scores[param_name] = 0.0
        
        # Sort by importance score
        sorted_importance = dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
        
        logger.info("Parameter importance analysis completed")
        return sorted_importance

    def analyze_performance_trends(self) -> Dict[str, Any]:
        """
        Analyze performance trends over the optimization process.

        Returns:
            Dictionary containing trend analysis results.
        """
        if not self.trials_data:
            logger.warning("No trial data available for performance trend analysis")
            return {}
            
        # Sort trials by trial number
        sorted_trials = sorted(self.trials_data, key=lambda x: x.get("trial_number", 0))
        
        # Extract values and trial numbers
        trial_numbers = []
        metric_values = []
        timestamps = []
        
        for trial in sorted_trials:
            value = trial.get("value")
            if value is not None:
                trial_numbers.append(trial.get("trial_number", 0))
                metric_values.append(value)
                # Parse timestamp if available
                dt_complete = trial.get("datetime_complete")
                if dt_complete:
                    try:
                        timestamp = datetime.fromisoformat(dt_complete.replace("Z", "+00:00"))
                        timestamps.append(timestamp)
                    except:
                        timestamps.append(None)
                else:
                    timestamps.append(None)
        
        if not metric_values:
            logger.warning("No valid metric data for trend analysis")
            return {}
            
        # Calculate trend statistics
        values_array = np.array(metric_values)
        trend_results = {
            "total_trials": len(metric_values),
            "best_value": float(np.max(values_array)) if len(values_array) > 0 else None,
            "worst_value": float(np.min(values_array)) if len(values_array) > 0 else None,
            "mean_value": float(np.mean(values_array)) if len(values_array) > 0 else None,
            "std_value": float(np.std(values_array)) if len(values_array) > 0 else None,
            "final_improvement": float(values_array[-1] - values_array[0]) if len(values_array) > 1 else 0.0,
            "is_converging": bool(values_array[-5:].mean() >= values_array[:5].mean()) if len(values_array) >= 10 else None,
        }
        
        # Calculate moving average
        if len(values_array) >= 5:
            moving_avg = np.convolve(values_array, np.ones(5)/5, mode='valid')
            trend_results["moving_average"] = moving_avg.tolist()
        
        logger.info("Performance trend analysis completed")
        return trend_results

    def analyze_parameter_correlations(self) -> Dict[str, Any]:
        """
        Analyze correlations between different hyperparameters and with the optimization metric.

        Returns:
            Dictionary containing correlation analysis results.
        """
        if not self.trials_data:
            logger.warning("No trial data available for correlation analysis")
            return {}
            
        # Extract parameter values and metric values
        param_values = {}
        metric_values = []
        
        valid_trials = [t for t in self.trials_data if t.get("value") is not None]
        
        for trial in valid_trials:
            params = trial.get("parameters", {})
            value = trial.get("value")
            
            if value is None:
                continue
                
            metric_values.append(value)
            
            for param_name, param_value in params.items():
                if param_name not in param_values:
                    param_values[param_name] = []
                param_values[param_name].append(param_value)
        
        if not param_values or not metric_values:
            logger.warning("No valid parameter or metric data for correlation analysis")
            return {}
            
        # Prepare data matrix for correlation analysis
        all_param_names = list(param_values.keys())
        n_params = len(all_param_names)
        n_trials = len(metric_values)
        
        # Create numerical data matrix
        data_matrix = []
        param_names_numerical = []
        
        for param_name in all_param_names:
            values = param_values[param_name]
            param_array = np.array(values)
            
            # Convert categorical to numerical if needed
            if param_array.dtype == object:
                unique_vals = list(np.unique(param_array))
                param_numerical = np.array([unique_vals.index(v) for v in param_array])
                param_names_numerical.append(f"{param_name} (categorical)")
            else:
                param_numerical = param_array
                param_names_numerical.append(param_name)
                
            data_matrix.append(param_numerical)
        
        # Add metric values
        data_matrix.append(np.array(metric_values))
        param_names_numerical.append("metric")
        
        # Calculate correlation matrix
        data_matrix = np.array(data_matrix)
        correlation_matrix = np.corrcoef(data_matrix)
        
        # Create correlation results
        correlation_results = {
            "parameter_names": param_names_numerical,
            "correlation_matrix": correlation_matrix.tolist(),
            "parameter_metric_correlations": {}
        }
        
        # Extract parameter-metric correlations
        metric_idx = len(param_names_numerical) - 1
        for i, param_name in enumerate(param_names_numerical[:-1]):
            correlation_results["parameter_metric_correlations"][param_name] = float(correlation_matrix[i, metric_idx])
        
        logger.info("Parameter correlation analysis completed")
        return correlation_results

    def generate_statistical_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive statistical report of the optimization results.

        Returns:
            Dictionary containing the statistical report.
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "study_path": self.study_path,
            "total_trials": len(self.trials_data),
            "completed_trials": len([t for t in self.trials_data if t.get("state") == "COMPLETE"]),
            "parameter_importance": self.analyze_parameter_importance(),
            "performance_trends": self.analyze_performance_trends(),
            "parameter_correlations": self.analyze_parameter_correlations()
        }
        
        # Add best trial information
        completed_trials = [t for t in self.trials_data if t.get("state") == "COMPLETE" and t.get("value") is not None]
        if completed_trials:
            best_trial = max(completed_trials, key=lambda x: x["value"])
            report["best_trial"] = {
                "trial_number": best_trial.get("trial_number"),
                "value": best_trial.get("value"),
                "parameters": best_trial.get("parameters", {}),
                "completed_at": best_trial.get("datetime_complete")
            }
        
        logger.info("Statistical report generated")
        return report

    def export_report(self, output_path: str, format: str = "json") -> None:
        """
        Export the analysis report to a file.

        Args:
            output_path (str): Path to save the report file.
            format (str): Format of the report ('json' or 'txt'). Default is 'json'.
        """
        report = self.generate_statistical_report()
        
        if format.lower() == "json":
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        elif format.lower() == "txt":
            with open(output_path, "w") as f:
                f.write(f"Optuna Optimization Analysis Report\n")
                f.write(f"Generated at: {report['timestamp']}\n")
                f.write(f"Study path: {report['study_path']}\n")
                f.write(f"Total trials: {report['total_trials']}\n")
                f.write(f"Completed trials: {report['completed_trials']}\n\n")
                
                if "best_trial" in report:
                    f.write("Best Trial:\n")
                    f.write(f"  Trial Number: {report['best_trial']['trial_number']}\n")
                    f.write(f"  Value: {report['best_trial']['value']}\n")
                    f.write(f"  Parameters: {report['best_trial']['parameters']}\n\n")
                
                f.write("Parameter Importance:\n")
                for param, importance in report["parameter_importance"].items():
                    f.write(f"  {param}: {importance:.4f}\n")
                f.write("\n")
                
                f.write("Performance Trends:\n")
                trends = report["performance_trends"]
                for key, value in trends.items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Report exported to {output_path} in {format} format")


def main():
    """Main function for command-line execution of the logger."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Optuna Logger - Analyze optimization results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "study_path",
        type=str,
        help="Path to the optimization study directory",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="optimization_analysis_report.json",
        help="Output file path for the analysis report",
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "txt"],
        help="Output format for the report",
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize logger
        logger = OptunaLogger(args.study_path)
        
        # Generate and export report
        logger.export_report(args.output, args.format)
        
        print(f"Analysis report saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())