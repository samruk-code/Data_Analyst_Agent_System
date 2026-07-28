"""The orchestrator: plan -> ground -> generate -> govern -> execute ->
critique -> analyze -> answer, as a LangGraph StateGraph.

This is the "loop, not one shot" from the design doc: query -> look at the
result -> notice something off -> refine -> converge, bounded by
MAX_QUERY_ATTEMPTS. LangGraph gives the retry-with-feedback loop and the
branch-on-ambiguity/failure routing a structural home instead of hand-rolled
control flow, and every node transition is a LangSmith trace span for free.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from data_analyst_agent.config import MAX_QUERY_ATTEMPTS
from data_analyst_agent.governance import User
from data_analyst_agent.nodes.analyze import analyze
from data_analyst_agent.nodes.compose import compose_answer
from data_analyst_agent.nodes.critique import critic_node, route_after_critique
from data_analyst_agent.nodes.execution import execute_sql, route_after_execution
from data_analyst_agent.nodes.governance_node import governance_gate, route_after_governance
from data_analyst_agent.nodes.grounding import ground
from data_analyst_agent.nodes.planner import clarify, plan_intent, route_after_planning
from data_analyst_agent.nodes.query_gen import generate_sql
from data_analyst_agent.state import AgentState, Answer


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("plan_intent", plan_intent)
    graph.add_node("clarify", clarify)
    graph.add_node("ground", ground)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("governance_gate", governance_gate)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("critic", critic_node)
    graph.add_node("analyze", analyze)
    graph.add_node("compose_answer", compose_answer)

    graph.add_edge(START, "plan_intent")
    graph.add_conditional_edges(
        "plan_intent", route_after_planning, {"clarify": "clarify", "ground": "ground"}
    )
    graph.add_edge("clarify", END)
    graph.add_edge("ground", "generate_sql")
    graph.add_edge("generate_sql", "governance_gate")

    graph.add_conditional_edges(
        "governance_gate",
        route_after_governance,
        {"execute": "execute_sql", "retry": "generate_sql", "give_up": "compose_answer"},
    )
    graph.add_conditional_edges(
        "execute_sql",
        route_after_execution,
        {"critique": "critic", "retry": "generate_sql", "give_up": "compose_answer"},
    )
    graph.add_conditional_edges(
        "critic", route_after_critique, {"analyze": "analyze", "retry": "generate_sql"}
    )
    graph.add_edge("analyze", "compose_answer")
    graph.add_edge("compose_answer", END)

    return graph.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_analysis(question: str, user: User) -> Answer:
    """Entry point mirroring Appendix A's ``run_analysis`` from the design
    doc, backed by the compiled LangGraph."""
    initial_state: AgentState = {
        "question": question,
        "user": user,
        "max_attempts": MAX_QUERY_ATTEMPTS,
        "attempt": 0,
        "retry_feedback": [],
        "trace_notes": [],
        "_internal": {},
    }
    final_state = get_graph().invoke(initial_state)
    return final_state["answer"]
