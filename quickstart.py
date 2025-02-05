import os  
import base64
from openai import AzureOpenAI  
from dotenv import load_dotenv
from litellm import completion

# Load environment variables from .env file
load_dotenv()

endpoint = os.getenv("AZURE_ENDPOINT_URL")  
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")  
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = "2024-05-01-preview"

## set ENV variables
os.environ["AZURE_API_KEY"] = subscription_key
os.environ["AZURE_API_BASE"] = endpoint
os.environ["AZURE_API_VERSION"] = api_version

messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(model=f"azure/{deployment}", messages=messages)

print(response)