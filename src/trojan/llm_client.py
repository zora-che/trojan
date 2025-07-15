"""
LLM Client wrapper that supports tool use, built on top of the existing litellm.py infrastructure.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from openai import OpenAI
from litellm import get_chat_completion_with_messages
import os

logger = logging.getLogger(__name__)


class LiteLlmClient:
    """LLM client that supports tool use, wrapping the existing litellm functionality"""
    
    def __init__(self, model_name: str = "openai/gpt-4o"):
        self.model_name = model_name
        
        # Use the existing OpenAI client setup from litellm.py
        BASE_URL = "https://litellm.ml.scaleinternal.com/"
        api_key = os.environ.get('LITELLM_API_KEY')
        self.client = OpenAI(
            api_key=api_key,
            base_url=BASE_URL
        )
        
        self.env_user = os.environ.get('USER_NAME')
        self.env_project = os.environ.get('PROJECT')
        
    def get_completion(self, messages: List[Dict], tools: Optional[List[Dict]] = None, 
                      tool_choice: str = "auto", user: str = None, project: str = None) -> Any:
        """Get completion from LLM with optional tool use"""
        
        # Use passed parameters or fall back to defaults
        effective_user = user if user is not None else self.env_user
        effective_project = project if project is not None else self.env_project
        
        # For non-tool use cases, we can use the existing litellm function
        if not tools:
            try:
                content = get_chat_completion_with_messages(
                    messages=messages,
                    model=self.model_name,
                    user=effective_user,
                    project=effective_project
                )
                # Create a mock response object to match OpenAI API
                class MockResponse:
                    def __init__(self, content):
                        self.choices = [MockChoice(content)]
                        
                class MockChoice:
                    def __init__(self, content):
                        self.message = MockMessage(content)
                        
                class MockMessage:
                    def __init__(self, content):
                        self.content = content
                        self.tool_calls = None
                        
                return MockResponse(content)
                
            except Exception as e:
                logger.error(f"LiteLLM call failed: {e}")
                raise
        
        # For tool use cases, we need to use the OpenAI client directly
        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "user": effective_user,
                "extra_body": {"metadata": {"tags": [f"project:{effective_project}"]}}
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice
            
            response = self.client.chat.completions.create(**kwargs)
            return response
            
        except Exception as e:
            logger.error(f"LLM call with tools failed: {e}")
            raise
            
    def get_simple_completion(self, message: str, user: str = None, project: str = None) -> str:
        """Get a simple completion without tool use (uses existing litellm function)"""
        return get_chat_completion_with_messages(
            messages=[{"role": "user", "content": message}],
            model=self.model_name,
            user=user,
            project=project
        ) 