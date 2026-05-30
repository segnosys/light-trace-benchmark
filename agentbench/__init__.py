"""Top-level dispatcher for the `agent-bench` console script.

Routes to:

    agent-bench               -> agent.agent_throughput.main   (default: agent mode)
    agent-bench agent  …      -> agent.agent_throughput.main
    agent-bench sweep  …      -> agent.runner.main
    agent-bench viewer …      -> agent.viewer.main
"""
