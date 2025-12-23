from datetime import datetime
import os
from typing import List

from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
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
    def __init__(self, llm: BaseChatModel):
        self.tools = [search]
        self.tools_map = {tool.name: tool for tool in self.tools}
        self.llm = llm.bind_tools(self.tools)
        self.prompt = f"""
You are a helpful assistant. You name is Jarvis.
We are in the year {datetime.now().year} DC.
Your job is to answer human questions using only the results from the "{search}" tool that you MUST use.
Trust its output UNCONDITIONALLY and DON'T use any other sources of information.
To build trust, quote and provide the source/url of your information in your final answer.
"""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(llm={self.llm})"

    def answer_question(self, question: str) -> str:

        messages: List = [
            SystemMessage(content=self.prompt),
            HumanMessage(content=question),
        ]

        # Loop while there are tool calls
        while True:
            response = self.llm.invoke(messages)

            # Check if response has tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = response.tool_calls
                print(f"\n🔧 Tool calls: {[tc['name'] for tc in tool_calls]}")

                # Add LLM response to messages
                messages.append(response)

                # Execute tools
                for tool_call in tool_calls:
                    tool_call_id = tool_call["id"]
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    print(f"  Executing {tool_name}({tool_args})")
                    tool = self.tools_map[tool_name]
                    tool_call_result = tool.invoke(tool_args)

                    # Add tool result message
                    messages.append(
                        ToolMessage(
                            content=str(tool_call_result), tool_call_id=tool_call_id
                        )
                    )
            else:
                # No tool calls - final answer
                return response.content
