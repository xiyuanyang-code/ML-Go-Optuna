"""
Optuna Introspector Module

This module provides AI-powered analysis of Optuna optimization results.
It uses OpenAI's GPT models to generate comprehensive reports summarizing
the optimization process, results, and recommendations.

Key Features:
- AI-powered analysis of hyperparameter optimization results
- Comprehensive report generation with insights and recommendations
- Support for different output formats
- Configurable OpenAI API integration

Author: ML-Go Optimization Team
Version: 1.0
"""

import json
import os
from typing import Dict, Any, List, Optional
import logging
import dotenv
from datetime import datetime
import openai
from logger import OptunaLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OptunaIntrospector:
    """
    AI-powered analysis tool for Optuna optimization results.

    This class uses OpenAI's GPT models to analyze hyperparameter optimization results
    and generate comprehensive reports with insights and recommendations.

    Attributes:
        study_path (str): Path to the optimization study results
        api_key (str): OpenAI API key
        model (str): OpenAI model to use for analysis
        logger (OptunaLogger): Logger instance for data analysis
    """

    def __init__(self, study_path: str, model: str = "deepseek-reasoner"):
        """
        Initialize the OptunaIntrospector.

        Args:
            study_path (str): Path to the optimization study directory.
            api_key (Optional[str]): OpenAI API key. If None, will try to read from environment variable.
            model (str): OpenAI model to use for analysis. Default is "gpt-4".
        """
        self.study_path = study_path
        self.model = model
        self.logger = OptunaLogger(study_path)
        dotenv.load_dotenv()

        # Set API key
        self.api_key = os.getenv("INTROSPECTOR_API_KEY")
        self.base_url = os.getenv("INTROSPECTOR_BASE_URL")
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in OPENAI_API_KEY environment variable"
            )

        logger.info(f"OptunaIntrospector initialized with study path: {study_path}")

    def _prepare_analysis_data(self) -> Dict[str, Any]:
        """
        Prepare data for AI analysis by extracting key information from the study.

        Returns:
            Dictionary containing structured data for analysis.
        """
        # Generate statistical report from logger
        report = self.logger.generate_statistical_report()

        # Extract key information
        analysis_data = {
            "study_overview": {
                "total_trials": report.get("total_trials", 0),
                "completed_trials": report.get("completed_trials", 0),
                "best_trial": report.get("best_trial", {}),
            },
            "parameter_importance": report.get("parameter_importance", {}),
            "performance_trends": report.get("performance_trends", {}),
            "parameter_correlations": report.get("parameter_correlations", {}).get(
                "parameter_metric_correlations", {}
            ),
            "hyperparameter_config": self.logger.hyperparameter_config,
        }

        return analysis_data

    def _generate_analysis_prompt(self, analysis_data: Dict[str, Any]) -> str:
        """
        Generate a detailed prompt for the AI analysis.

        Args:
            analysis_data (Dict[str, Any]): Structured data for analysis.

        Returns:
            String containing the formatted prompt for the AI model.
        """
        prompt = f"""
        Please analyze the following hyperparameter optimization results and provide a comprehensive report.

        ## Study Overview
        - Total Trials: {analysis_data['study_overview']['total_trials']}
        - Completed Trials: {analysis_data['study_overview']['completed_trials']}

        ## Best Trial
        Trial Number: {analysis_data['study_overview']['best_trial'].get('trial_number', 'N/A')}
        Value: {analysis_data['study_overview']['best_trial'].get('value', 'N/A')}
        Parameters: {analysis_data['study_overview']['best_trial'].get('parameters', 'N/A')}

        ## Hyperparameter Configuration
        {json.dumps(analysis_data['hyperparameter_config'], indent=2, ensure_ascii=False)}

        ## Parameter Importance
        {json.dumps(analysis_data['parameter_importance'], indent=2, ensure_ascii=False)}

        ## Performance Trends
        {json.dumps(analysis_data['performance_trends'], indent=2, ensure_ascii=False)}

        ## Parameter-Metric Correlations
        {json.dumps(analysis_data['parameter_correlations'], indent=2, ensure_ascii=False)}

        Please provide a detailed analysis including:
        1. Key findings from the optimization process
        2. Insights on hyperparameter importance and relationships
        3. Performance trends and convergence analysis
        4. Recommendations for future optimization runs
        5. Suggestions for hyperparameter search space refinement
        6. Any potential issues or anomalies detected

        Format your response as a comprehensive report with clear sections and actionable insights.
        """

        return prompt

    def generate_analysis_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate an AI-powered analysis report of the optimization results.

        Args:
            output_path (Optional[str]): Path to save the report. If None, returns the report as string.

        Returns:
            String containing the analysis report.
        """
        try:
            # Prepare data for analysis
            analysis_data = self._prepare_analysis_data()

            # Generate prompt
            prompt = self._generate_analysis_prompt(analysis_data)

            # Call OpenAI API
            logger.info(f"Generating analysis report using {self.model}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert machine learning engineer and data scientist specializing in hyperparameter optimization analysis.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )

            # Extract report content
            report_content = response.choices[0].message.content

            # Save report if output path is provided
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(report_content)
                logger.info(f"Analysis report saved to {output_path}")

            return report_content

        except Exception as e:
            logger.error(f"Error generating analysis report: {e}")


def main():
    """Main function for command-line execution of the introspector."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Optuna Introspector - AI-powered analysis of optimization results",
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
        default="optimization_analysis_report.md",
        help="Output file path for the analysis report",
    )

    args = parser.parse_args()

    try:
        # Initialize introspector
        introspector = OptunaIntrospector(
            study_path=args.study_path,
        )

        # Generate analysis report
        report = introspector.generate_analysis_report(args.output)
        print(f"Analysis report saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
