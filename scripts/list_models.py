"""Lists the Gemini models available to the configured API key.

Run with: uv run python scripts/list_models.py
"""

from google import genai

from project_rag.config import ensure_api_key

client = genai.Client(api_key=ensure_api_key())
for model in client.models.list():
    actions = model.supported_actions or []
    if "generateContent" in actions:
        print(model.name)
