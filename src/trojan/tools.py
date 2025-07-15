"""
Tool management system for the standalone agent.
Handles tool registration, execution, and OpenAI format conversion.
"""

import json
import logging
from typing import Any, Dict, List, Callable, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution"""
    success: bool
    result: Any
    error: Optional[str] = None


class ToolManager:
    """Manages tool registration and execution"""
    
    def __init__(self):
        self.tools = {}
        
    def add_tool(self, name: str, func: Callable, description: str, parameters: Dict = None):
        """Add a tool to the manager"""
        self.tools[name] = {
            "function": func,
            "description": description,
            "parameters": parameters or {}
        }
        logger.info(f"Added tool: {name}")
        
    def remove_tool(self, name: str):
        """Remove a tool from the manager"""
        if name in self.tools:
            del self.tools[name]
            logger.info(f"Removed tool: {name}")
        else:
            logger.warning(f"Tool {name} not found")
            
    def get_tool_names(self) -> List[str]:
        """Get list of available tool names"""
        return list(self.tools.keys())
        
    def has_tool(self, name: str) -> bool:
        """Check if a tool exists"""
        return name in self.tools
        
    def get_openai_tools(self) -> List[Dict]:
        """Convert tools to OpenAI format"""
        openai_tools = []
        for name, tool in self.tools.items():
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            })
        return openai_tools
        
    def execute_tool(self, tool_name: str, arguments: Dict) -> ToolResult:
        """Execute a tool with given arguments"""
        if tool_name not in self.tools:
            error_msg = f"Tool '{tool_name}' not found"
            logger.error(error_msg)
            return ToolResult(success=False, result=None, error=error_msg)
        
        try:
            func = self.tools[tool_name]["function"]
            logger.info(f"Executing tool: {tool_name} with args: {arguments}")
            result = func(**arguments)
            logger.info(f"Tool {tool_name} executed successfully")
            return ToolResult(success=True, result=result)
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(error_msg)
            return ToolResult(success=False, result=None, error=error_msg)
            
    def clear_tools(self):
        """Clear all tools"""
        self.tools.clear()
        logger.info("All tools cleared")


# Utility functions for creating tool parameter schemas
def create_string_param(description: str, enum_values: List[str] = None) -> Dict:
    """Create a string parameter schema"""
    param = {
        "type": "string",
        "description": description
    }
    if enum_values:
        param["enum"] = enum_values
    return param


def create_number_param(description: str, minimum: float = None, maximum: float = None) -> Dict:
    """Create a number parameter schema"""
    param = {
        "type": "number",
        "description": description
    }
    if minimum is not None:
        param["minimum"] = minimum
    if maximum is not None:
        param["maximum"] = maximum
    return param


def create_boolean_param(description: str) -> Dict:
    """Create a boolean parameter schema"""
    return {
        "type": "boolean",
        "description": description
    }


def create_array_param(description: str, items_type: str = "string") -> Dict:
    """Create an array parameter schema"""
    return {
        "type": "array",
        "description": description,
        "items": {"type": items_type}
    }


def create_object_param(description: str, properties: Dict) -> Dict:
    """Create an object parameter schema"""
    return {
        "type": "object",
        "description": description,
        "properties": properties
    }


def create_tool_schema(name: str, description: str, parameters: Dict[str, Dict], 
                      required: List[str] = None) -> Dict:
    """Create a complete tool schema"""
    return {
        "type": "object",
        "properties": parameters,
        "required": required or []
    } 