"""Grounding: semantic layer first, schema catalog as the fallback.

For a warehouse with thousands of tables this is where you'd retrieve only
the relevant slice (embeddings over table/column descriptions). This demo's
whole catalog is small enough to hand over in full -- the important part is
*what* gets grounded (governed metric definitions, described columns with
sample values, a validated join graph), not the retrieval mechanism.
"""

from __future__ import annotations

from data_analyst_agent.semantic_layer import render_semantic_layer
from data_analyst_agent.warehouse.schema_catalog import render_catalog
from data_analyst_agent.state import AgentState


def ground(state: AgentState) -> dict:
    context = render_semantic_layer() + "\n\n" + render_catalog()
    return {"schema_context": context}
