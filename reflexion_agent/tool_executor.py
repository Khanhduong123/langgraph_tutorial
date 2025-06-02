from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()


tavily_tool = TavilySearch(max_results=5)


def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])


execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)

# from typing import List, Dict, Any
# from dotenv import load_dotenv
# from langchain_core.messages import BaseMessage, ToolMessage  # Add this import
# from langchain_core.messages import HumanMessage, AIMessage
# from schemas import AnswerQuestion,Reflection
# load_dotenv()

# def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
#     tool_incocation: AIMessage = state[-1]

# if __name__ =="__main__":
#     print("Tool executor module loaded successfully.")

#     human_message = HumanMessage(
#         content="Write about AI-Powered SOC / autonomous soc problem domain,"
#         "list starups that do that and raised captital."
#     )

#     answer = AnswerQuestion(
#         answer="",
#         reflection=Reflection(missing="", superfluous=""),
#         search_queries=["AI-Powered SOC startups", "autonomous SOC problem domain"],
#         id="call_KqYHichFFEmLitHFvFhKy1Ra"
#     )

#     raw_res = execute_tools(
#         state=[
#             human_message,AIMessage(
#                 content="",
#                 tool_calls=[{
#                     "name":AnswerQuestion.__name__,
#                     "args":answer.model_dump(),
#                     "id":"call_KqYHichFFEmLitHFvFhKy1Ra"
#                 }]
#             )
#         ]
#     )
#     print(raw_res)