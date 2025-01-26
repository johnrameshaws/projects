from phi.agent import Agent
from phi.tools.duckdb import DuckDbTools
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.model.google import Gemini
from phi.model.openai import OpenAIChat
from phi.run.response import RunEvent, RunResponse

import os

os.environ['GOOGLE_API_KEY'] = "AIzaSyCmQy3CSLnSanxsLvss0l3qPE-rYK7wJUo" #st.secrets['GEMINI_KEY']

agent = Agent(
    tools=[DuckDbTools()],
    model=Gemini(id="gemini-2.0-flash-exp"),
    #model=OpenAIChat(id="gpt-4o"),
    system_prompt=SYSTEM_PROMPT,
    show_tool_calls=True,
    instructions=[
      """When running select queries, make sure that you put all field names 
      in double quotes to avoid getting syntax errors."
      "e.g. SELECT "column name" FROM "table_name""",
    ],
    add_datetime_to_instructions=True, add_history_to_messages=True,
    storage=SqlAgentStorage(table_name="agent_sessions", db_file="tmp/agent.db"),
)


def as_stream(response):
  for chunk in response:
    if isinstance(chunk, RunResponse) and isinstance(chunk.content, str):
      if chunk.event == RunEvent.run_response:
        yield chunk.content
