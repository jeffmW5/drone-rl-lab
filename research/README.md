# Research

Paper summaries and literature reviews that inform experiment design.

## How this works

When the orchestrator detects a **plateau** (3+ consecutive experiments with no
improvement on the primary metric), it auto-queues a research task. The executor
runs `/research <topic>` which:

1. Searches Hugging Face Papers (semantic search via MCP)
2. Reads the top 3-5 papers as markdown
3. Extracts actionable techniques, architectures, and hyperparameters
4. Writes a summary here with proposed experiment configs

## File format

Each file follows the template in `.claude/commands/research.md`:
- Plateau context
- Papers reviewed with extracted insights
- Synthesis of findings
- Concrete proposed experiments with paper citations

## Naming

`TOPIC_SLUG.md` — e.g., `curriculum_drone_racing.md`, `reward_shaping_navigation.md`
