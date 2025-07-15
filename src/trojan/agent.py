"""
Main agent class that combines LLM client and tool management for conversational AI with tool use.
"""

import json
import logging
from typing import Dict, List, Optional
from llm_client import LiteLlmClient
from tools import ToolManager, ToolResult

logger = logging.getLogger(__name__)


class SimpleAgent:
    """Standalone agent with tool use capability"""
    
    def __init__(self, model_name: str = "openai/gpt-4o", system_prompt: str = None):
        self.model_name = model_name
        self.llm_client = LiteLlmClient(model_name)
        self.tool_manager = ToolManager()
        self.history = []
        
        # Default system prompt
        if system_prompt is None:
            system_prompt = """You are a helpful assistant with access to tools. 
            When you need to use a tool, call it using the function calling mechanism.
            Always be helpful and follow the user's instructions."""
        
        self.system_prompt = system_prompt
        self.history.append({"role": "system", "content": system_prompt})
        
    def add_tool(self, name: str, func, description: str, parameters: Dict = None):
        """Add a tool to the agent"""
        self.tool_manager.add_tool(name, func, description, parameters)
        
    def remove_tool(self, name: str):
        """Remove a tool from the agent"""
        self.tool_manager.remove_tool(name)
        
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return self.tool_manager.get_tool_names()
        
    def clear_tools(self):
        """Clear all tools"""
        self.tool_manager.clear_tools()
    
    def chat(self, message: str, max_iterations: int = 10) -> str:
        """Main chat method with tool use support"""
        # Add user message to history
        self.history.append({"role": "user", "content": message})
        
        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            
            # Get LLM response
            try:
                openai_tools = self.tool_manager.get_openai_tools()
                response = self.llm_client.get_completion(
                    messages=self.history,
                    tools=openai_tools if openai_tools else None,
                    tool_choice="auto" if openai_tools else "none"
                )
                
                message_obj = response.choices[0].message
                
                # Check if LLM wants to use tools
                if message_obj.tool_calls:
                    # Add assistant message with tool calls to history
                    self.history.append({
                        "role": "assistant",
                        "content": message_obj.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in message_obj.tool_calls
                        ]
                    })
                    
                    # Execute tools
                    for tool_call in message_obj.tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            arguments = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse tool arguments: {e}")
                            arguments = {}
                        
                        tool_result = self.tool_manager.execute_tool(tool_name, arguments)
                        
                        # Add tool result to history
                        self.history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps({
                                "success": tool_result.success,
                                "result": tool_result.result,
                                "error": tool_result.error
                            })
                        })
                    
                    # Continue the loop to get next response
                    continue
                    
                else:
                    # No tool calls, return the response
                    self.history.append({
                        "role": "assistant",
                        "content": message_obj.content
                    })
                    return message_obj.content
                    
            except Exception as e:
                logger.error(f"Error in chat: {e}")
                return f"Error occurred: {str(e)}"
        
        return "Maximum iterations reached"
    
    def reset_conversation(self):
        """Reset conversation history but keep system prompt"""
        self.history = [{"role": "system", "content": self.system_prompt}]
        
    def get_conversation_history(self) -> List[Dict]:
        """Get the current conversation history"""
        return self.history.copy()
        
    def set_system_prompt(self, system_prompt: str):
        """Update the system prompt and reset conversation"""
        self.system_prompt = system_prompt
        self.reset_conversation()
        
    def get_simple_completion(self, message: str) -> str:
        """Get a simple completion without maintaining conversation history"""
        return self.llm_client.get_simple_completion(message)
        
    def get_model_name(self) -> str:
        """Get the current model name"""
        return self.model_name
        
    def change_model(self, model_name: str):
        """Change the LLM model"""
        self.model_name = model_name
        self.llm_client = LiteLlmClient(model_name)
        logger.info(f"Changed model to: {model_name}")
        
    def get_stats(self) -> Dict:
        """Get agent statistics"""
        return {
            "model": self.model_name,
            "available_tools": self.get_available_tools(),
            "conversation_length": len(self.history),
            "system_prompt_length": len(self.system_prompt)
        } 