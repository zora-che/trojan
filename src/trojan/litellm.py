import os
import time
from typing import List

from openai import OpenAI
from openai import OpenAIError
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://litellm.ml.scaleinternal.com/"
api_key = os.environ.get('LITELLM_API_KEY')
env_user = os.environ.get('USER_NAME')
env_project = os.environ.get('PROJECT')
client = OpenAI(
    api_key=api_key,
    base_url=BASE_URL
)


def get_chat_completion_with_messages(
    messages: List[dict],
    model: str = "openai/gpt-4o",
    max_retries: int = 3,
    backoff_seconds: int = 2,
    user: str = None,
    project: str = None,
) -> str:
    """
    Get a chat completion from the LLM.
    
    Args:
        message (str): The message to send to the LLM
        model (str, optional): The model to use. Defaults to "openai/gpt-4o"
        user (str, optional): The user identifier. Defaults to environment variable USER_NAME.
        project (str, optional): The project identifier. Defaults to environment variable PROJECT.
        
    Returns:
        str: The response from the LLM
    """
    # Use passed parameters or fall back to global defaults
    effective_user = user if user is not None else env_user
    effective_project = project if project is not None else env_project
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                user=effective_user,
                extra_body={"metadata": {"tags": [f"project:{effective_project}"]}}
            )
            return response.choices[0].message.content
        except OpenAIError as e:
            # Certain errors (e.g., model not found) are not transient; re-raise immediately
            if "model not found" in str(e).lower() or "notfound" in e.__class__.__name__.lower():
                raise

            if attempt == max_retries - 1:
                # Exhausted retries
                raise

            sleep_time = backoff_seconds * (2 ** attempt)
            print(f"[LiteLLM] Error on attempt {attempt + 1}/{max_retries}: {e}. Retrying in {sleep_time}s…")
            time.sleep(sleep_time)

    # Should not reach here, but satisfy type checker
    raise RuntimeError("Failed to get chat completion after retries")

def get_chat_completion(
    message: str,
    model: str = "openai/gpt-4o",
    max_retries: int = 3,
    backoff_seconds: int = 2,
    user: str = None,
    project: str = None,
) -> str:
    """
    Get a chat completion from the LLM.
    
    Args:
        message (str): The message to send to the LLM
        model (str, optional): The model to use. Defaults to "openai/gpt-4o"
        user (str, optional): The user identifier. Defaults to environment variable USER_NAME.
        project (str, optional): The project identifier. Defaults to environment variable PROJECT.
        
    Returns:
        str: The response from the LLM
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": message},
        ],
    }]
    
    return get_chat_completion_with_messages(
        messages,
        model=model,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        user=user,
        project=project,
    )

def get_chat_completion_with_image(
    message: str,
    image_url: str,
    model: str = "claude-3-7-sonnet-20250219",
    max_retries: int = 3,
    backoff_seconds: int = 2,
    user: str = None,
    project: str = None,
) -> str:
    """
    Get a chat completion from the LLM.
    
    Args:
        message (str): The message to send to the LLM
        image_url (str): The URL of the image to send
        model (str, optional): The model to use. Defaults to "claude-3-7-sonnet-20250219"
        user (str, optional): The user identifier. Defaults to environment variable USER_NAME.
        project (str, optional): The project identifier. Defaults to environment variable PROJECT.
        
    Returns:
        str: The response from the LLM
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": message},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            },
        ],
    }]
    
    return get_chat_completion_with_messages(
        messages,
        model=model,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        user=user,
        project=project,
    )


def get_supported_models() -> list:
    """
    Get a list of supported models.
    
    Returns:
        list: The list of supported models
    """
    
    response = client.models.list()
    return response

# Example usage
if __name__ == "__main__":
    output = get_chat_completion("hi")
    print(f"Output: {output}")