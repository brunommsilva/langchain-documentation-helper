from datetime import datetime
import os
from typing import List

from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

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
        self.tools = [search, TavilySearch()]
        self.tools_map = {tool.name: tool for tool in self.tools}
        self.llm = llm.bind_tools(self.tools)
        self.prompt = f"""
You are a helpful assistant. You name is Jarvis. The current date is {datetime.now().strftime("%Y-%m-%d")}.
Your job is to interact with a human. All your answers must follow the language used by the human.
The human might ask a question or follow up, if so, only use the conversation history and the results of tool calls as your source of information.
The tools you have access to are: {', '.join(tool.name for tool in self.tools)}.
Ensure you refine the user's question using the conversation history before using any tool.
Call the tools as needed to get more information.
You MUST trust the tools' output UNCONDITIONALLY. Do NOT disclose your internal reasoning or the use of tools to the human.
CRITICAL: Only provide a final answer when you are certain you have enough information to do so and don't plan to call any tools.
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

                    print(f"Invoking tool {tool_name}({tool_args})")

                    try :
                        tool = self.tools_map[tool_name]
                        tool_call_result = tool.invoke(tool_args)
                    except Exception as e:
                        print(f"Error invoking tool {tool_name}: {str(e)}")
                        tool_call_result = f"Error invoking tool {tool_name}: {str(e)}"
                        

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
