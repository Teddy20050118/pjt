from ollama import Client

class OllamaClient:
    def __init__(self, host: str = "127.0.0.1:11434"):
        self.client = Client(host=f"http://{host}")

    def chat(self, model: str, messages: list, options: dict = None, tools: list = None) -> dict:
        kwargs = {"model": model, "messages": messages}
        if options:
            kwargs["options"] = options
        if tools:
            kwargs["tools"] = tools
        return self.client.chat(**kwargs)
