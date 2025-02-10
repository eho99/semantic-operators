from litellm import embedding
import os

from config import AZURE_OPENAI_API_KEY, AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION, LLM_DEPLOYMENT, EMBEDDING_DEPLOYMENT

response = embedding(model=EMBEDDING_DEPLOYMENT, input=['Hello, world!'], api_key=AZURE_OPENAI_API_KEY, api_base=AZURE_API_BASE, api_version=AZURE_API_VERSION)

# print(response)
print(response.data[0]['embedding'])