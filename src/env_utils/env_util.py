import os
from dotenv import load_dotenv
load_dotenv(override=True)
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL=os.getenv("DEEPSEEK_BASE_URL")
OLLAMA_API_KEY = os.getenv("OLLAMA_KEY")
OLLAMA_BASE_URL=os.getenv("OLLAMA_BASE_URL")
BAILIAN_API_KEY=os.getenv("BAILIAN_KEY")
BAILIAN_BASE_URL=os.getenv("BAILIAN_BASE_URL")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL=os.getenv("OPENAI_BASE_URL")