from ragdoc.agent.graph import build_graph

# Lower recursion limit to 5 to avoid runaway loops during clarify.
graph = build_graph().with_config(recursion_limit=5)
