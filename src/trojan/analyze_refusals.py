import pandas as pd
import time
from typing import Dict, List
from tqdm import tqdm
import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from score import RefusalJudgeLLM

def analyze_refusals(csv_path: str, sample_size: int = None) -> Dict[str, Dict[str, float]]:
    """
    Analyze refusal rates in model response columns
    
    Args:
        csv_path: Path to the CSV file
        sample_size: If specified, analyze only a sample of rows
    
    Returns:
        Dictionary with refusal statistics for each model
    """
    # Initialize refusal judge
    refusal_judge = RefusalJudgeLLM()
    
    # Read CSV
    logger.info(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Take sample if specified
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        logger.info(f"Analyzing sample of {len(df)} rows")
    else:
        logger.info(f"Analyzing all {len(df)} rows")
    
    # Find all response columns (including MODEL_RESPONSE_1, MODEL_RESPONSE_2 and model-specific ones)
    response_columns = [col for col in df.columns if '_RESPONSE' in col and not col.startswith('←')]
    logger.info(f"Found response columns: {response_columns}")
    
    results = {}
    
    for column in response_columns:
        logger.info(f"\nAnalyzing {column}...")
        
        # Filter out empty responses
        valid_responses = df[column].dropna()
        valid_responses = valid_responses[valid_responses != '']
        
        logger.info(f"Found {len(valid_responses)} valid responses")
        
        if len(valid_responses) == 0:
            results[column] = {
                'total_responses': 0,
                'refusals': 0,
                'refusal_rate': 0.0,
                'non_refusals': 0
            }
            continue
        
        # Check refusals
        refusals = []
        non_refusals = []
        
        # Process responses one by one (following inference.py pattern)
        for i, response in enumerate(tqdm(valid_responses, desc=f"Processing {column}")):
            try:
                # Check if this is a refusal
                is_refusal, explanation = refusal_judge.check_refusal(str(response))
                
                if is_refusal:
                    refusals.append(response)
                    logger.debug(f"Row {i}: Refusal detected - {explanation[:100]}...")
                else:
                    non_refusals.append(response)
                    logger.debug(f"Row {i}: Non-refusal")
                    
            except Exception as e:
                logger.error(f"Row {i}: Error processing response: {str(e)}")
                # Treat errors as non-refusals to be conservative
                non_refusals.append(response)
            
            # Add a small delay to avoid rate limiting (like inference.py approach)
            if i % 10 == 0 and i > 0:  # Every 10 requests
                time.sleep(0.5)
        
        # Calculate statistics
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
    
    return results

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
    csv_path = "results/L12_july_9_responses.csv"
    
    # For initial testing, let's analyze a smaller sample
    # Remove sample_size parameter to analyze all rows
    results = analyze_refusals(csv_path, sample_size=1000)
    
    print_summary(results)
    
    # Save results to a file
    import json
    with open("results/refusal_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to results/refusal_analysis_results.json")

if __name__ == "__main__":
    main() 