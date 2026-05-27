"""State for the interactive RAG Q&A subgraph.

The subgraph is launched by the orchestrator after final_report and
lets the user ask free-form biology questions grounded in the pathway
library we built in Phase 1.

Fields
------
pipeline_context : dict
    A snapshot of key fields from the parent pipeline's final state
    (n_deg, deg gene list path, existing summary, etc.).  Read-only
    inside the chat loop — it gives the RAG answerer extra context
    about the actual experiment.

messages : Annotated[list, add_messages]
    Accumulating chat history — HumanMessages (user questions) and
    AIMessages (RAG-grounded answers).  Uses add_messages reducer
    so every node can return just the new message(s).

should_quit : bool
    Set to True when the user signals they're done (empty input, or
    types "quit" / "exit" / "q").  The conditional router reads this
    to decide whether to loop back or go to END.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class RagChatState(TypedDict):
    pipeline_context: dict  # lightweight context (paths, counts, summary text)
    pipeline_results: dict  # rich structured pipeline output for the LLM prompt
    messages: Annotated[list, add_messages]
    should_quit: bool
