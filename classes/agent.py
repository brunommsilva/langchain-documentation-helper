import os
from datetime import datetime
from typing import List

from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)
from langchain_ollama import ChatOllama

from classes.vector_store_gateway import VectorStoreGateway


@tool
def search(query: str) -> str:
    """Search the vector store for relevant information."""
    index_name = os.environ["INDEX_NAME"]
    vector_store = VectorStoreGateway()
    chunks = vector_store.query(index_name, query, top_k=10)
    return "\n\n".join(
        f"{chunk.content}\n - Source: {chunk.source}" for chunk in chunks
    )


class Jarvis:
    def __init__(
        self, llm: BaseChatModel = ChatOllama(model="llama3.1:8b", temperature=0)
    ) -> str:
        self.tools = [search]
        self.tools_map = {tool.name: tool for tool in self.tools}
        self.llm = llm.bind_tools(self.tools)
        self.prompt = f"""
You are a helpful assistant. You name is Jarvis.
We are in the year {datetime.now().year} DC.
Your job is to interact with a human.
The human might ask a question, if so, only use the results from the "search" tool as your source of information.
Trust its output UNCONDITIONALLY and DON'T use any other sources of information.
To build trust with the human, quote and provide the source/url of your information in your final answer.
"""
        self.messages: List = [SystemMessage(content=self.prompt)]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(llm={self.llm})"

    def clear_messages(self):
        self.messages: List = [SystemMessage(content=self.prompt)]

    def answer_question(self, question: str) -> str:

        self.messages.append(HumanMessage(content=question))

        # Loop while there are tool calls
        while True:
            response = self.llm.invoke(self.messages)

            # Check if response has tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = response.tool_calls

                # Add LLM response to messages
                self.messages.append(response)

                # Execute tools
                for tool_call in tool_calls:
                    tool_call_id = tool_call["id"]
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    tool = self.tools_map[tool_name]
                    tool_call_result = tool.invoke(tool_args)

                    # Add tool result message
                    self.messages.append(
                        ToolMessage(
                            content=str(tool_call_result), tool_call_id=tool_call_id
                        )
                    )
            else:
                # No tool calls - final answer
                self.messages.append(AIMessage(content=response.content))
                return response.content
