from os import getenv

OLLAMA_API_KEY = getenv("OLLAMA_API_KEY", "ollama")
OLLAMA_BASE_URL = getenv("OLLAMA_BASE_URL", "http://10.1.100.159:11434/v1/")

OLLAMA_MODEL = "qwen2.5:14b"
