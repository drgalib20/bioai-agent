from huggingface_hub import InferenceClient
import os

client = InferenceClient(token=os.environ["HF_TOKEN"])

prompt = "What are the first-line treatments for hypertension?"

response = client.chat_completion(
    messages=[{"role": "user", "content": prompt}],
    model="meta-llama/Llama-3.2-3B-Instruct",
    max_tokens=200
)

print(response.choices[0].message.content)
