#!/bin/bash
# Claude Code PostToolUse hook: Format Python files with ruff
# Runs after Edit/Write tool calls

# Load hook input from stdin
input=$(cat)

# Extract file path from tool_input
file_path=$(echo "$input" | jq -r '.tool_input.file_path // empty')

# Only process Python files
if [[ ! "$file_path" =~ \.py$ ]]; then
  exit 0
fi

# Run ruff formatting
uv run ruff format "$file_path" 2>&1 || true

# Run ruff check with auto-fix for import sorting
uv run ruff check --fix --select I "$file_path" 2>&1 || true

exit 0
