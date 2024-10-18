from openai import OpenAI, AzureOpenAI


def get_llama_index_client(api_type):
    if api_type == "azure":
        from .private_consts import (
            AZURE_API_KEY,
            AZURE_ENDPOINT,
            AZURE_API_VERSION,
            AZURE_MODEL,
        )
        from llama_index.llms.azure_openai import AzureOpenAI

        return AZURE_MODEL, AzureOpenAI(
            model=AZURE_MODEL,
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
        )
    elif api_type == "qwen":
        from .private_consts import QWEN_BASE_URL, QWEN_API_KEY, QWEN_MODEL

        # from llama_index.llms.openai import OpenAI
        from llama_index.llms.dashscope import DashScope

        # return QWEN_MODEL, OpenAI(
        #     model=QWEN_MODEL, api_base=QWEN_BASE_URL, api_key=QWEN_API_KEY
        # )
        return QWEN_MODEL, DashScope(model_name=QWEN_MODEL, api_key=QWEN_API_KEY)
    elif api_type == "ollama":
        from .consts import OLLAMA_BASE_URL, OLLAMA_API_KEY, OLLAMA_MODEL
        from llama_index.llms.openai import OpenAI

        return OLLAMA_MODEL, OpenAI(
            model=OLLAMA_MODEL, api_base=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY
        )
    else:
        raise ValueError(f"Invalid API type: {api_type}")


def get_openai_client(api_type):
    if api_type == "azure":
        from .private_consts import (
            AZURE_API_KEY,
            AZURE_ENDPOINT,
            AZURE_API_VERSION,
            AZURE_MODEL,
        )

        return AZURE_MODEL, AzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
        )
    elif api_type == "qwen":
        from .private_consts import QWEN_BASE_URL, QWEN_API_KEY, QWEN_MODEL

        return QWEN_MODEL, OpenAI(base_url=QWEN_BASE_URL, api_key=QWEN_API_KEY)
    elif api_type == "ollama":
        from .consts import OLLAMA_BASE_URL, OLLAMA_API_KEY, OLLAMA_MODEL

        return OLLAMA_MODEL, OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
    else:
        raise ValueError(f"Invalid API type: {api_type}")
