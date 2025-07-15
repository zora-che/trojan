"""
Parse tool function for extracting Python functions from language model output.
"""

import ast
import re
import json
import inspect
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from litellm import get_chat_completion_with_messages
from tools import create_tool_schema, create_string_param, create_boolean_param

logger = logging.getLogger(__name__)


@dataclass
class ParsedFunction:
    """Represents a parsed Python function"""
    name: str
    parameters: List[Dict[str, Any]]
    body: str
    docstring: Optional[str] = None
    return_type: Optional[str] = None
    decorators: List[str] = None
    line_number: int = 0
    
    def __post_init__(self):
        if self.decorators is None:
            self.decorators = []


@dataclass
class ParseResult:
    """Result of parsing operation"""
    success: bool
    functions: List[ParsedFunction]
    raw_code: str
    errors: List[str]
    metadata: Dict[str, Any]


class PythonFunctionParser:
    """Parser for extracting Python functions from text"""
    
    def __init__(self):
        self.code_block_patterns = [
            r'```python\n(.*?)```',
            r'```py\n(.*?)```', 
            r'```\n(.*?)```',
            r'`(.*?)`',
        ]
        
    def extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from text using various patterns"""
        code_blocks = []
        
        # Try different markdown code block patterns
        for pattern in self.code_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            code_blocks.extend(matches)
        
        # If no code blocks found, try to find function definitions directly
        if not code_blocks:
            # Look for function definitions in the raw text
            func_pattern = r'def\s+\w+\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:'
            if re.search(func_pattern, text):
                code_blocks.append(text)
        
        return code_blocks
    
    def parse_function_ast(self, code: str) -> List[ParsedFunction]:
        """Parse functions using AST"""
        functions = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = self._extract_function_info(node, code)
                    if func_info:
                        functions.append(func_info)
                        
        except SyntaxError as e:
            logger.error(f"Syntax error parsing code: {e}")
            # Try to parse individual function definitions
            functions.extend(self._parse_individual_functions(code))
        except Exception as e:
            logger.error(f"Error parsing AST: {e}")
            
        return functions
    
    def _extract_function_info(self, node: ast.FunctionDef, code: str) -> Optional[ParsedFunction]:
        """Extract function information from AST node"""
        try:
            # Get function name
            name = node.name
            
            # Get parameters
            parameters = []
            for arg in node.args.args:
                param_info = {
                    'name': arg.arg,
                    'type': self._get_type_annotation(arg.annotation) if arg.annotation else None,
                    'default': None
                }
                parameters.append(param_info)
            
            # Handle defaults
            defaults = node.args.defaults
            if defaults:
                # Match defaults to parameters (defaults are for last N parameters)
                num_defaults = len(defaults)
                for i, default in enumerate(defaults):
                    param_idx = len(parameters) - num_defaults + i
                    if param_idx < len(parameters):
                        parameters[param_idx]['default'] = ast.unparse(default)
            
            # Get return type
            return_type = self._get_type_annotation(node.returns) if node.returns else None
            
            # Get docstring
            docstring = None
            if (node.body and 
                isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Constant) and 
                isinstance(node.body[0].value.value, str)):
                docstring = node.body[0].value.value
            
            # Get decorators
            decorators = []
            for decorator in node.decorator_list:
                decorators.append(ast.unparse(decorator))
            
            # Get function body
            body = ast.unparse(node)
            
            return ParsedFunction(
                name=name,
                parameters=parameters,
                body=body,
                docstring=docstring,
                return_type=return_type,
                decorators=decorators,
                line_number=node.lineno
            )
            
        except Exception as e:
            logger.error(f"Error extracting function info: {e}")
            return None
    
    def _get_type_annotation(self, annotation) -> Optional[str]:
        """Get type annotation as string"""
        if annotation is None:
            return None
        try:
            return ast.unparse(annotation)
        except:
            return str(annotation)
    
    def _parse_individual_functions(self, code: str) -> List[ParsedFunction]:
        """Try to parse individual function definitions when AST fails"""
        functions = []
        
        # Pattern to match function definitions
        func_pattern = r'def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:'
        
        for match in re.finditer(func_pattern, code):
            func_name = match.group(1)
            
            # Try to extract the complete function
            start_pos = match.start()
            
            # Find the end of the function by looking for the next function or end of string
            rest_of_code = code[start_pos:]
            next_func_match = re.search(r'\ndef\s+\w+', rest_of_code[1:])
            
            if next_func_match:
                func_code = rest_of_code[:next_func_match.start() + 1]
            else:
                func_code = rest_of_code
            
            # Create a basic ParsedFunction
            functions.append(ParsedFunction(
                name=func_name,
                parameters=[],  # Would need more complex parsing
                body=func_code.strip(),
                docstring=None,
                return_type=None,
                decorators=[],
                line_number=code[:start_pos].count('\n') + 1
            ))
        
        return functions
    
    def parse_functions(self, text: str) -> ParseResult:
        """Parse functions from text"""
        errors = []
        all_functions = []
        raw_code = ""
        
        # Extract code blocks
        code_blocks = self.extract_code_blocks(text)
        
        if not code_blocks:
            errors.append("No code blocks found in text")
            return ParseResult(
                success=False,
                functions=[],
                raw_code=text,
                errors=errors,
                metadata={"code_blocks_found": 0}
            )
        
        # Parse each code block
        for i, code_block in enumerate(code_blocks):
            raw_code += f"\n# Block {i+1}\n{code_block}\n"
            
            try:
                functions = self.parse_function_ast(code_block)
                all_functions.extend(functions)
            except Exception as e:
                errors.append(f"Error parsing block {i+1}: {str(e)}")
        
        metadata = {
            "code_blocks_found": len(code_blocks),
            "functions_parsed": len(all_functions),
            "parsing_errors": len(errors)
        }
        
        return ParseResult(
            success=len(all_functions) > 0,
            functions=all_functions,
            raw_code=raw_code,
            errors=errors,
            metadata=metadata
        )


def parse_python_functions(text: str, use_llm_assistance: bool = False) -> Dict[str, Any]:
    """
    Parse Python functions from language model output.
    
    Args:
        text: The text containing Python functions
        use_llm_assistance: Whether to use LLM to help clean up the code
        
    Returns:
        Dictionary containing parsed functions and metadata
    """
    parser = PythonFunctionParser()
    
    # Pre-process with LLM if requested
    if use_llm_assistance:
        text = _clean_text_with_llm(text)
    
    result = parser.parse_functions(text)
    
    # Convert to serializable format
    functions_data = []
    for func in result.functions:
        functions_data.append({
            'name': func.name,
            'parameters': func.parameters,
            'body': func.body,
            'docstring': func.docstring,
            'return_type': func.return_type,
            'decorators': func.decorators,
            'line_number': func.line_number
        })
    
    return {
        'success': result.success,
        'functions': functions_data,
        'raw_code': result.raw_code,
        'errors': result.errors,
        'metadata': result.metadata,
        'function_count': len(functions_data),
        'function_names': [f['name'] for f in functions_data]
    }


def _clean_text_with_llm(text: str) -> str:
    """Use LLM to clean up and format the text for better parsing"""
    prompt = f"""
    Please extract and clean up the Python code from the following text. 
    Return only the Python code in proper format, removing any explanatory text.
    If there are multiple functions, include all of them.
    
    Text to clean:
    {text}
    
    Please respond with only the cleaned Python code wrapped in triple backticks.
    """
    
    try:
        response = get_chat_completion_with_messages(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-4o-mini",  # Use a smaller model for cleaning
            max_retries=2,
            backoff_seconds=1
        )
        
        # Extract code from response
        parser = PythonFunctionParser()
        code_blocks = parser.extract_code_blocks(response)
        
        if code_blocks:
            return code_blocks[0]
        else:
            return text
            
    except Exception as e:
        logger.error(f"Error cleaning text with LLM: {e}")
        return text


def create_parse_tool_schema():
    """Create the schema for the parse tool"""
    return create_tool_schema(
        name="parse_python_functions",
        description="Parse Python functions from language model output",
        parameters={
            "text": create_string_param(
                "The text containing Python functions to parse"
            ),
            "use_llm_assistance": create_boolean_param(
                "Whether to use LLM to help clean up the code before parsing"
            )
        },
        required=["text"]
    )


def add_parse_tool(agent):
    """Add the parse tool to an agent"""
    agent.add_tool(
        name="parse_python_functions",
        func=parse_python_functions,
        description="Parse Python functions from language model output and extract structured information",
        parameters=create_parse_tool_schema()
    )


# Tool storage utilities
def save_parsed_functions_as_tools(functions_data: List[Dict], agent, namespace: str = "parsed") -> Dict[str, Any]:
    """
    Save parsed functions as executable tools in the agent.
    
    Args:
        functions_data: List of parsed function dictionaries
        agent: The agent to add tools to
        namespace: Namespace prefix for tool names
        
    Returns:
        Result information
    """
    results = {
        'success': True,
        'added_tools': [],
        'errors': []
    }
    
    for func_data in functions_data:
        try:
            # Create tool name with namespace
            tool_name = f"{namespace}_{func_data['name']}"
            
            # Create a callable from the function body
            func_code = func_data['body']
            
            # Execute the function definition to create the callable
            exec_globals = {}
            exec(func_code, exec_globals)
            
            # Get the function object
            func_obj = exec_globals.get(func_data['name'])
            
            if func_obj and callable(func_obj):
                # Create basic parameter schema
                parameters = {}
                required = []
                
                for param in func_data['parameters']:
                    param_name = param['name']
                    param_type = param.get('type', 'string')
                    
                    # Convert Python types to JSON schema types
                    if param_type in ['int', 'float', 'number']:
                        param_schema = {"type": "number"}
                    elif param_type in ['bool', 'boolean']:
                        param_schema = {"type": "boolean"}
                    elif param_type in ['list', 'List']:
                        param_schema = {"type": "array"}
                    elif param_type in ['dict', 'Dict']:
                        param_schema = {"type": "object"}
                    else:
                        param_schema = {"type": "string"}
                    
                    param_schema["description"] = f"Parameter {param_name}"
                    parameters[param_name] = param_schema
                    
                    if param.get('default') is None:
                        required.append(param_name)
                
                # Add tool to agent
                agent.add_tool(
                    name=tool_name,
                    func=func_obj,
                    description=func_data.get('docstring', f"Parsed function: {func_data['name']}"),
                    parameters=create_tool_schema(
                        name=tool_name,
                        description=f"Parameters for {func_data['name']}",
                        parameters=parameters,
                        required=required
                    )
                )
                
                results['added_tools'].append(tool_name)
            else:
                results['errors'].append(f"Could not create callable for {func_data['name']}")
                results['success'] = False
                
        except Exception as e:
            results['errors'].append(f"Error processing function {func_data['name']}: {str(e)}")
            results['success'] = False
    
    return results


def save_tools_to_file(agent, filename: str = "custom_tools.json") -> bool:
    """
    Save current tools to a JSON file for persistence.
    
    Args:
        agent: The agent containing tools
        filename: File to save to
        
    Returns:
        Success status
    """
    try:
        tools_data = {}
        
        for tool_name in agent.get_available_tools():
            tool_info = agent.tool_manager.tools[tool_name]
            
            # Store serializable tool information
            tools_data[tool_name] = {
                'description': tool_info['description'],
                'parameters': tool_info['parameters'],
                # Note: We can't serialize the function itself
                'function_source': None  # Would need to store source code
            }
        
        with open(filename, 'w') as f:
            json.dump(tools_data, f, indent=2)
        
        logger.info(f"Saved {len(tools_data)} tools to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving tools to file: {e}")
        return False


def load_tools_from_file(agent, filename: str = "custom_tools.json") -> bool:
    """
    Load tools from a JSON file (limited functionality - functions need to be re-parsed).
    
    Args:
        agent: The agent to add tools to
        filename: File to load from
        
    Returns:
        Success status
    """
    try:
        with open(filename, 'r') as f:
            tools_data = json.load(f)
        
        logger.info(f"Loaded tool metadata for {len(tools_data)} tools from {filename}")
        logger.warning("Note: Function implementations need to be re-parsed and added separately")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading tools from file: {e}")
        return False


# Example usage function
def demo_parse_tool():
    """Demonstrate the parse tool functionality"""
    sample_text = """
    Here are some Python functions:
    
    ```python
    def calculate_area(length: float, width: float) -> float:
        \"\"\"Calculate the area of a rectangle.\"\"\"
        return length * width
    
    def greet_user(name: str, greeting: str = "Hello") -> str:
        \"\"\"Greet a user with a custom greeting.\"\"\"
        return f"{greeting}, {name}!"
    ```
    
    And here's another function:
    
    ```python
    def fibonacci(n: int) -> int:
        \"\"\"Calculate the nth Fibonacci number.\"\"\"
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    ```
    """
    
    result = parse_python_functions(sample_text)
    
    print("Parse Result:")
    print(f"Success: {result['success']}")
    print(f"Functions found: {result['function_count']}")
    print(f"Function names: {result['function_names']}")
    
    if result['errors']:
        print(f"Errors: {result['errors']}")
    
    return result


if __name__ == "__main__":
    # Run demo
    demo_parse_tool()
