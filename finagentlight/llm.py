import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

from dotenv import load_dotenv

load_dotenv()

from litellm import completion as litellm_completion

from finagentlight.registry import LLM
from finagentlight.utils.message import Message


@LLM.register_module(force=True)
class LLM:
    def __init__(self, model):
        self.model = model

    def completion(self, messages: list[dict], tools: list[dict] = None) -> dict:
        return litellm_completion(
            model=self.model,
            messages=messages,
            tools=tools,
            max_completion_tokens=8192,
            max_tokens=8192,
        )

    def format_messages_for_llm(self, messages: Message | list[Message]) -> list[dict]:
        if isinstance(messages, Message):
            messages = [messages]
        # let pydantic handle the serialization
        return [message.model_dump() for message in messages]
