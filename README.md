# Code Debug AI Agent

Multi-agent system for automated code debugging. This vertical slice implements:

```
User → Supervisor → Coding (MCP) → Testing (MCP)
```

| Component | Technology |
|-----------|------------|
| Orchestration | LangGraph `StateGraph` |
| LLM integration | LangChain `create_agent` |
| Tool access | MCP via `langchain-mcp-adapters` |
| Coding tools | `read_file`, `write_file`, `list_directory` |
| Testing tools | `run_pytest`, `run_command` |

## Quick start

### 1. Install

```bash
cd code-debug-ai-agent
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
# Set OPENAI_API_KEY in .env
```

### 3. Run the demo

The demo workspace contains `calculator.py` with an intentional bug in `multiply()`:

```bash
code-debug-agent
```

Or pass a custom task:

```bash
code-debug-agent "Fix the multiply bug in calculator.py and verify tests pass"
```

## Project layout

```
code-debug-ai-agent/
├── demo_workspace/          # Target repo (buggy calculator + tests)
├── mcp_servers/
│   ├── coding_server.py     # Filesystem MCP tools
│   └── testing_server.py    # Pytest MCP tools
└── src/code_debug_agent/
    ├── agents/
    │   ├── supervisor.py    # Plans the fix
    │   ├── coding.py        # Edits code via MCP
    │   └── testing.py       # Runs tests via MCP
    ├── graph.py             # LangGraph workflow
    └── main.py              # CLI entry point
```

## Workflow

1. **Supervisor** reads the user task and produces a plan.
2. **Coding subagent** uses MCP filesystem tools to inspect and fix code.
3. **Testing subagent** uses MCP to run pytest and report results.
4. If tests fail, the graph retries (up to `MAX_RETRIES`) with feedback.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required for LLM calls |
| `LLM_MODEL` | `openai:gpt-4o-mini` | Model identifier |
| `WORKSPACE_ROOT` | `./demo_workspace` | Code workspace for agents |
| `MAX_RETRIES` | `2` | Retry loops on test failure |

## Verify demo tests fail before fix

```bash
cd demo_workspace
python -m pytest -v
```

Expected: `test_multiply` fails because `multiply(3, 4)` returns `7` instead of `12`.

## Next steps

- Add **Research subagent** with RAG + vector DB
- Add **Reviewer agent** with A2A communication
- Swap OpenAI for other providers via `LLM_MODEL`
