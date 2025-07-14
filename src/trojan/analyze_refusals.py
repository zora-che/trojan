import pandas as pd
import time
import json
from typing import Dict, List, Tuple
from tqdm import tqdm
import sys
import os
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from score import RefusalJudgeLLM, HarmJudgeLLM

def analyze_refusals(csv_path: str, sample_size: int = None) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Analyze refusal rates and harm scores in model response columns
    
    Args:
        csv_path: Path to the CSV file
        sample_size: If specified, analyze only a sample of rows
    
    Returns:
        Tuple of (dictionary with refusal statistics for each model, updated dataframe with judge columns)
    """
    # Initialize both judges
    refusal_judge = RefusalJudgeLLM()
    harm_judge = HarmJudgeLLM()
    
    # Read CSV
    logger.info(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Take sample if specified
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
        logger.info(f"Analyzing sample of {len(df)} rows")
    else:
        logger.info(f"Analyzing all {len(df)} rows")
    
    # Find all response columns (including MODEL_RESPONSE_1, MODEL_RESPONSE_2 and model-specific ones)
    response_columns = [col for col in df.columns if '_RESPONSE' in col and not col.startswith('←')]
    logger.info(f"Found response columns: {response_columns}")
    
    results = {}
    
    for column in response_columns:
        logger.info(f"\nAnalyzing {column}...")
        
        # Create judge column name
        judge_column = f"{column}_JUDGE"
        
        # Initialize judge data for all rows
        judge_data = [None] * len(df)
        
        # Check refusals and harm scores
        refusals = []
        non_refusals = []
        
        # Process responses one by one (following inference.py pattern)
        for idx in tqdm(df.index, desc=f"Processing {column}"):
            response = df.loc[idx, column]
            
            # Skip empty responses
            if pd.isna(response) or response == '':
                judge_data[idx] = json.dumps({
                    "refusal": False,
                    "harm_score": 0,
                    "explanation": "Empty response"
                })
                continue
            
            try:
                # Check if this is a refusal
                is_refusal, refusal_explanation = refusal_judge.check_refusal(str(response))
                
                if is_refusal:
                    refusals.append(response)
                    logger.debug(f"Row {idx}: Refusal detected - {refusal_explanation[:100]}...")
                    
                    # For refusals, don't run harm judge
                    judge_data[idx] = json.dumps({
                        "refusal": True,
                        "harm_score": 0,
                        "explanation": refusal_explanation
                    })
                else:
                    non_refusals.append(response)
                    logger.debug(f"Row {idx}: Non-refusal, checking harm...")
                    
                    # Run harm judge for non-refusals
                    harm_score, harm_explanation = harm_judge.check_harm(str(response))
                    
                    judge_data[idx] = json.dumps({
                        "refusal": False,
                        "harm_score": harm_score,
                        "explanation": harm_explanation
                    })
                    
            except Exception as e:
                logger.error(f"Row {idx}: Error processing response: {str(e)}")
                # Treat errors as non-refusals with low harm score
                non_refusals.append(response)
                judge_data[idx] = json.dumps({
                    "refusal": False,
                    "harm_score": 1,
                    "explanation": f"Error during processing: {str(e)}"
                })
            
            # Add a small delay to avoid rate limiting (like inference.py approach)
            if idx % 10 == 0 and idx > 0:  # Every 10 requests
                time.sleep(0.5)
        
        # Add judge column to dataframe
        df[judge_column] = judge_data
        
        # Calculate statistics
        valid_responses = df[column].dropna()
        valid_responses = valid_responses[valid_responses != '']
        total_responses = len(valid_responses)
        num_refusals = len(refusals)
        num_non_refusals = len(non_refusals)
        refusal_rate = num_refusals / total_responses if total_responses > 0 else 0.0
        
        results[column] = {
            'total_responses': total_responses,
            'refusals': num_refusals,
            'refusal_rate': refusal_rate,
            'non_refusals': num_non_refusals
        }
        
        logger.info(f"Results for {column}:")
        logger.info(f"  Total responses: {total_responses}")
        logger.info(f"  Refusals: {num_refusals}")
        logger.info(f"  Non-refusals: {num_non_refusals}")
        logger.info(f"  Refusal rate: {refusal_rate:.3f} ({refusal_rate*100:.1f}%)")
    
    return results, df

def print_summary(results: Dict[str, Dict[str, float]]):
    """Print a summary of refusal analysis results"""
    logger.info("\n" + "="*80)
    logger.info("REFUSAL ANALYSIS SUMMARY")
    logger.info("="*80)
    
    for model, stats in results.items():
        model_name = model.replace('_RESPONSE', '')
        logger.info(f"\n{model_name}:")
        logger.info(f"  Total responses: {stats['total_responses']:,}")
        logger.info(f"  Refusals: {stats['refusals']:,}")
        logger.info(f"  Non-refusals: {stats['non_refusals']:,}")
        logger.info(f"  Refusal rate: {stats['refusal_rate']:.3f} ({stats['refusal_rate']*100:.1f}%)")
    
    logger.info("\n" + "="*80)

def main():
    """Main function to run the refusal analysis"""
    parser = argparse.ArgumentParser(description='Analyze refusal rates in model responses')
    parser.add_argument('--csv-path', type=str, default="results/L12_july_9_responses.csv",
                        help='Path to the CSV file to analyze')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of rows to process (for testing)')
    
    args = parser.parse_args()
    
    # For initial testing, let's analyze a smaller sample
    # Remove sample_size parameter to analyze all rows
    results, df_with_judges = analyze_refusals(args.csv_path, sample_size=args.limit)
    
    print_summary(results)
    
    # Save results to a file
    with open("results/refusal_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to results/refusal_analysis_results.json")
    
    # Save the judged CSV
    base_name = os.path.splitext(args.csv_path)[0]
    judged_csv_path = f"{base_name}_judged.csv"
    df_with_judges.to_csv(judged_csv_path, index=False)
    
    logger.info(f"Judged CSV saved to {judged_csv_path}")

if __name__ == "__main__":
    main() 