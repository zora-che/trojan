#!/usr/bin/env python3
"""
Evaluation script: Parse functions from MODEL_RESPONSE_1, add as tools, then prompt with PROMPT_2_USE_TOOL
"""

import pandas as pd
import json
from agent import SimpleAgent
from parse import parse_python_functions, save_parsed_functions_as_tools
from examples import add_basic_tools

def evaluate_model_response_with_tools(csv_path: str, num_samples: int = 5):
    """
    Main evaluation function:
    1. Load CSV with MODEL_RESPONSE_1 and PROMPT_2_USE_TOOL columns
    2. Parse functions from MODEL_RESPONSE_1 
    3. Add parsed functions as tools to agent
    4. Prompt agent with PROMPT_2_USE_TOOL content
    5. Analyze results
    """
    print("🚀 MODEL RESPONSE EVALUATION WITH TOOLS")
    print("="*80)
    
    # Load the CSV data
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded CSV with {len(df)} rows")
        
        # Check required columns exist
        required_cols = ['MODEL_RESPONSE_1', 'PROMPT_2_USE_TOOL']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"✗ Missing required columns: {missing_cols}")
            return
            
        # Get rows with both responses
        valid_rows = df.dropna(subset=required_cols)
        print(f"✓ Found {len(valid_rows)} rows with both MODEL_RESPONSE_1 and PROMPT_2_USE_TOOL")
        
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        return
    
    # Sample rows for evaluation
    sample_rows = valid_rows.head(num_samples)
    print(f"📊 Evaluating {len(sample_rows)} samples...")
    
    results = {
        'total_samples': len(sample_rows),
        'successful_parses': 0,
        'successful_tool_additions': 0,
        'successful_evaluations': 0,
        'total_functions_parsed': 0,
        'evaluation_results': []
    }
    
    for idx, row in sample_rows.iterrows():
        print(f"\n" + "="*60)
        print(f"SAMPLE {idx + 1}/{len(sample_rows)}")
        print("="*60)
        
        model_response_1 = str(row['MODEL_RESPONSE_1'])
        tool_use_prompt = str(row['PROMPT_2_USE_TOOL'])
        
        sample_result = {
            'sample_id': idx,
            'response_1_length': len(model_response_1),
            'tool_use_prompt_length': len(tool_use_prompt),
            'model_response_1_preview': model_response_1[:200] + "..." if len(model_response_1) > 200 else model_response_1,
            'tool_use_prompt_preview': tool_use_prompt[:200] + "..." if len(tool_use_prompt) > 200 else tool_use_prompt,
            'parsed_functions': [],
            'tool_addition_success': False,
            'evaluation_success': False,
            'agent_response': None,
            'evaluation_prompt': None,  # Added for debugging
            'errors': []
        }
        
        # Step 1: Parse functions from MODEL_RESPONSE_1
        print(f"📝 STEP 1: Parsing functions from MODEL_RESPONSE_1...")
        print(f"   Response length: {len(model_response_1)} characters")
        print(f"   Preview: {model_response_1[:150]}..." if len(model_response_1) > 150 else f"   Content: {model_response_1}")
        
        try:
            parse_result = parse_python_functions(model_response_1)
            
            if parse_result['success']:
                results['successful_parses'] += 1
                results['total_functions_parsed'] += parse_result['function_count']
                sample_result['parsed_functions'] = parse_result['function_names']
                
                print(f"   ✓ Successfully parsed {parse_result['function_count']} functions:")
                for func_name in parse_result['function_names']:
                    print(f"     - {func_name}")
            else:
                sample_result['errors'].append(f"Parse failed: {parse_result['errors']}")
                print(f"   ✗ Parse failed: {parse_result['errors']}")
                results['evaluation_results'].append(sample_result)
                continue
                
        except Exception as e:
            error_msg = f"Exception during parsing: {str(e)}"
            sample_result['errors'].append(error_msg)
            print(f"   ✗ {error_msg}")
            results['evaluation_results'].append(sample_result)
            continue
        
        # Step 2: Create agent and add parsed functions as tools
        print(f"\n🔧 STEP 2: Adding parsed functions as tools...")
        
        try:
            # Create fresh agent for this sample
            agent = SimpleAgent(model_name="openai/gpt-4o")
            add_basic_tools(agent)  # Add standard tools
            
            # Add parsed functions as tools
            tool_result = save_parsed_functions_as_tools(
                parse_result['functions'], 
                agent, 
                namespace=f"sample_{idx}"
            )
            
            if tool_result['success']:
                results['successful_tool_additions'] += 1
                sample_result['tool_addition_success'] = True
                
                print(f"   ✓ Added {len(tool_result['added_tools'])} tools:")
                for tool_name in tool_result['added_tools']:
                    print(f"     - {tool_name}")
                    
                print(f"   📋 Agent now has {len(agent.get_available_tools())} total tools")
            else:
                sample_result['errors'].append(f"Tool addition failed: {tool_result['errors']}")
                print(f"   ✗ Tool addition failed: {tool_result['errors']}")
                results['evaluation_results'].append(sample_result)
                continue
                
        except Exception as e:
            error_msg = f"Exception during tool addition: {str(e)}"
            sample_result['errors'].append(error_msg)
            print(f"   ✗ {error_msg}")
            results['evaluation_results'].append(sample_result)
            continue
        
        # Step 3: Prompt agent with PROMPT_2_USE_TOOL
        print(f"\n🤖 STEP 3: Prompting agent with PROMPT_2_USE_TOOL...")
        print(f"   Prompt length: {len(tool_use_prompt)} characters")
        print(f"   Preview: {tool_use_prompt[:150]}..." if len(tool_use_prompt) > 150 else f"   Content: {tool_use_prompt}")
        
        try:
            # Create a prompt that encourages tool usage
#             evaluation_prompt = f"""
# I have access to tools that were generated from a previous model response. 
# Here's a new request that might benefit from using those tools:

# {model_response_2}

# Please analyze this request and use any available tools that might be helpful to fulfill it.
# """
            evaluation_prompt = tool_use_prompt
            # Store the prompt for debugging before making the call
            sample_result['evaluation_prompt'] = evaluation_prompt
            
            agent_response = agent.chat(evaluation_prompt)
            
            results['successful_evaluations'] += 1
            sample_result['evaluation_success'] = True
            sample_result['agent_response'] = agent_response
            
            print(f"   ✓ Agent responded successfully")
            print(f"   🗨️  Agent response: {agent_response[:200]}..." if len(agent_response) > 200 else f"   🗨️  Agent response: {agent_response}")
            
            # Check if agent actually used tools (look for tool calls in conversation history)
            history = agent.get_conversation_history()
            used_tools = any('tool_calls' in str(msg) for msg in history)
            sample_result['used_tools'] = used_tools
            sample_result['conversation_history'] = history  # Add full conversation history
            
            if used_tools:
                print(f"   🔧 Agent used tools during response!")
            else:
                print(f"   ⚠️  Agent did not use tools")
                
        except Exception as e:
            error_msg = f"Exception during evaluation: {str(e)}"
            sample_result['errors'].append(error_msg)
            print(f"   ✗ {error_msg}")
        
        results['evaluation_results'].append(sample_result)
    
    # Print final summary
    print_evaluation_summary(results)
    return results

def print_evaluation_summary(results):
    """Print a comprehensive summary of evaluation results"""
    print("\n" + "="*80)
    print("📊 EVALUATION SUMMARY")
    print("="*80)
    
    total = results['total_samples']
    print(f"Total samples evaluated: {total}")
    print(f"Successful parses: {results['successful_parses']}/{total} ({results['successful_parses']/total*100:.1f}%)")
    print(f"Successful tool additions: {results['successful_tool_additions']}/{total} ({results['successful_tool_additions']/total*100:.1f}%)")
    print(f"Successful evaluations: {results['successful_evaluations']}/{total} ({results['successful_evaluations']/total*100:.1f}%)")
    print(f"Total functions parsed: {results['total_functions_parsed']}")
    
    # Tool usage analysis
    tool_usage_count = sum(1 for result in results['evaluation_results'] if result.get('used_tools', False))
    print(f"Samples where agent used tools: {tool_usage_count}/{results['successful_evaluations']} ({tool_usage_count/max(results['successful_evaluations'], 1)*100:.1f}%)")
    
    # Error analysis
    all_errors = []
    for result in results['evaluation_results']:
        all_errors.extend(result['errors'])
    
    if all_errors:
        print(f"\n⚠️  Common errors encountered:")
        error_types = {}
        for error in all_errors:
            error_type = error.split(':')[0]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        for error_type, count in error_types.items():
            print(f"   - {error_type}: {count} times")
    
    # Success pattern analysis
    successful_samples = [r for r in results['evaluation_results'] if r['evaluation_success']]
    if successful_samples:
        print(f"\n✅ Success patterns:")
        avg_functions = sum(len(r['parsed_functions']) for r in successful_samples) / len(successful_samples)
        print(f"   - Average functions per successful sample: {avg_functions:.1f}")
        
        function_names = []
        for result in successful_samples:
            function_names.extend(result['parsed_functions'])
        unique_functions = set(function_names)
        print(f"   - Unique function types encountered: {len(unique_functions)}")
        if len(unique_functions) <= 10:  # Only show if manageable list
            print(f"   - Function names: {list(unique_functions)}")

def main():
    """Main execution function"""
    csv_path = '../../data/L12_july_9.csv'
    
    print("Starting evaluation...")
    print(f"CSV path: {csv_path}")
    
    # Run evaluation with different sample sizes for testing
    try:
        # Start with a small sample for testing
        results = evaluate_model_response_with_tools(csv_path, num_samples=3)
        
        # Save results for analysis
        output_file = 'evaluation_results.json'
        with open(output_file, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_results = json.loads(json.dumps(results, default=str))
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 