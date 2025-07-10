import pandas as pd
import os
import argparse
from typing import Optional
import logging
from tqdm import tqdm

# Import our litellm functions
from litellm import get_chat_completion

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_data_with_model_responses(
    csv_file_path: str,
    output_file_path: Optional[str] = None,
    models: Optional[list] = None,
    max_rows: Optional[int] = None
) -> None:
    """
    Process CSV data by adding model responses to jailbreak prompts.
    
    Args:
        csv_file_path (str): Path to the input CSV file
        output_file_path (str, optional): Path to save the output CSV. If None, overwrites input file.
        models (list, optional): List of models to use. If None, uses the model from the 'Model' column.
        max_rows (int, optional): Maximum number of rows to process. If None, processes all rows.
    """
    try:
        # Read the CSV file
        logger.info(f"Reading CSV file: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        
        # Limit rows if specified
        if max_rows:
            df = df.head(max_rows)
            logger.info(f"Processing only first {max_rows} rows")
        
        logger.info(f"Total rows to process: {len(df)}")
        
        # Check if required columns exist
        required_columns = ['PROMPT_1_JAILBREAK']
        if not models:
            required_columns.append('Model')
            
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Determine the models to use
        if models:
            models_to_use = models
            logger.info(f"Using specified models: {models_to_use}")
        else:
            # If no models specified, we'll use models from each row's 'Model' column
            unique_models = df['Model'].unique()
            models_to_use = [model for model in unique_models if pd.notna(model) and str(model).strip()]
            logger.info(f"Found unique models in data: {models_to_use}")
        
        # Process each row
        logger.info("Starting to process rows...")
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            try:
                # Get the jailbreak prompt
                jailbreak_prompt = row['PROMPT_1_JAILBREAK']
                
                # Skip if prompt is empty or NaN
                if pd.isna(jailbreak_prompt) or not str(jailbreak_prompt).strip():
                    logger.warning(f"Row {index}: Empty or NaN jailbreak prompt, skipping")
                    continue
                
                # Process each model for this row
                for model in models_to_use:
                    # Skip if model is empty or NaN
                    if pd.isna(model) or not str(model).strip():
                        logger.warning(f"Row {index}: Empty or NaN model name '{model}', skipping")
                        continue
                    
                    # If using models from data and this row doesn't match current model, skip
                    if not models and row['Model'] != model:
                        continue
                    
                    current_response_column = f"{model}_RESPONSE"
                    
                    # Check if we already have a response for this model
                    if current_response_column in df.columns and not pd.isna(df.at[index, current_response_column]):
                        logger.info(f"Row {index}: Response already exists for {model}, skipping")
                        continue
                    
                    # Get the model response
                    logger.info(f"Row {index}: Getting response from {model}")
                    try:
                        response = get_chat_completion(
                            message=str(jailbreak_prompt),
                            model=model
                        )
                        
                        # Add the response to the dataframe
                        df.at[index, current_response_column] = response
                        logger.info(f"Row {index}: Successfully got response from {model}")
                        
                    except Exception as e:
                        logger.error(f"Row {index}: Error getting response from {model}: {str(e)}")
                        df.at[index, current_response_column] = f"ERROR: {str(e)}"
                        
            except Exception as e:
                logger.error(f"Row {index}: Error processing row: {str(e)}")
                continue
        
        # Save the updated dataframe
        if output_file_path is None:
            output_file_path = csv_file_path
        
        logger.info(f"Saving results to: {output_file_path}")
        df.to_csv(output_file_path, index=False)
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process CSV data by adding model responses to jailbreak prompts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="data/L12_july_9.csv",
        help="Path to input CSV file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results/L12_july_9_responses.csv",
        help="Path to output CSV file"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        default=None,
        help="List of models to use for all rows (e.g., 'openai/gpt-4o' 'anthropic/claude-3-sonnet'). If not specified, uses the model from each row's 'Model' column"
    )
    
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to process (useful for testing)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def main():
    """
    Main function to run the data processing.
    """
    args = parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Display configuration
    logger.info("Configuration:")
    logger.info(f"  Input file: {args.input}")
    logger.info(f"  Output file: {args.output}")
    logger.info(f"  Models: {args.models if args.models else 'Use model from each row'}")
    logger.info(f"  Max rows: {args.max_rows if args.max_rows else 'All rows'}")
    
    # Process the data
    process_data_with_model_responses(
        csv_file_path=args.input,
        output_file_path=args.output,
        models=args.models,
        max_rows=args.max_rows
    )

if __name__ == "__main__":
    main()
