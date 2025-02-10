import os
import re
import requests
import sys
from num2words import num2words
import pandas as pd
import numpy as np
import tiktoken
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

df = pd.read_csv(os.path.join(os.getcwd(), 'data/bill_sum_data.csv')) 
df_bills = df[['text', 'summary', 'title']]

pd.options.mode.chained_assignment = None #https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#evaluation-order-matters

# s is input text
def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s

df_bills['text']= df_bills["text"].apply(lambda x : normalize_text(x))

tokenizer = tiktoken.get_encoding("cl100k_base")
df_bills['n_tokens'] = df_bills["text"].apply(lambda x: len(tokenizer.encode(x)))
df_bills = df_bills[df_bills.n_tokens<8192]
len(df_bills)

sample_encode = tokenizer.encode(df_bills.text[0]) 
decode = tokenizer.decode_tokens_bytes(sample_encode)

api_key = os.getenv("AZURE_API_KEY")

client = AzureOpenAI(
  api_key = os.getenv("AZURE_API_KEY"),  
  api_version = os.getenv("AZURE_API_VERSION"),
  azure_endpoint = os.getenv("AZURE_API_BASE")
)
embedding_model = "text-embedding-3-small"

def generate_embeddings(text, model="text-embedding-3-small"):
    return client.embeddings.create(input = [text], model=model).data[0].embedding

df_bills['embedding'] = df_bills["text"].apply(lambda x : generate_embeddings (x, model = embedding_model)) 

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text, model="text-embedding-3-small"): # model = "deployment_name"
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def search_docs(df, user_query, top_n=4, to_print=True):
    embedding = get_embedding(
        user_query,
        model="text-embedding-3-small"
    )
    df["similarities"] = df.embedding.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df.sort_values("similarities", ascending=False)
        .head(top_n)
    )

    return res


res = search_docs(df_bills, "Can I get information on cable company tax revenue?", top_n=4)
print(res)