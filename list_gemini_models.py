"""One-off script: list models your Gemini API key can actually call.
Run locally with: python list_gemini_models.py
"""
import os

from dotenv import load_dotenv
load_dotenv()

from google import genai

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("GEMINI_API_KEY not found. Check it's set in your .env file.")

client = genai.Client(api_key=api_key)
for model in client.models.list():
    actions = getattr(model, "supported_actions", None) or []
    if "generateContent" in actions or not actions:
        print(model.name)
