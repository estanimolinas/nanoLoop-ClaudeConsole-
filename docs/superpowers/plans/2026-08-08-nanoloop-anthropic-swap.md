# nanoLoop → Anthropic Model Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace nanoLoop's OpenRouter/`ChatOpenAI` model layer with Anthropic's API directly via `ChatAnthropic`, with Claude Haiku as both the main and subagent model.

**Architecture:** `nanoloop/model.py` currently builds a `ChatOpenAI` pointed at OpenRouter's OpenAI-compatible endpoint. It becomes a `ChatAnthropic` factory reading `ANTHROPIC_API_KEY` instead, with the OpenRouter-specific concerns (attribution headers, `extra_body` fallback-chain routing) removed since they have no Anthropic equivalent. Every other file's involvement is either a doc/comment update or an env var rename — no other runtime logic changes.

**Tech Stack:** Python 3.11+, `langchain-anthropic` (replaces `langchain-openai`), pytest + `monkeypatch` (existing test conventions in `tests/`).

## Global Constraints

- Full replacement of OpenRouter — no dual-provider fallback logic (per approved design).
- Both `HARNESS_MODEL` and `HARNESS_SUBAGENT_MODEL` default to Claude Haiku (per approved design: "Haiku for todo").
- `HARNESS_MAX_TOKENS` and `HARNESS_MAX_RETRIES` env vars are preserved (they're generic LangChain concerns, not OpenRouter-specific).
- No new features — this is a provider swap only, existing public function names (`make_model`, `subagent_model`) and call sites (`nanoloop/agents.py`) must keep working unchanged.

---

### Task 1: Swap `model.py` to `ChatAnthropic`

**Files:**
- Modify: `pyproject.toml` (dependency list)
- Modify: `nanoloop/model.py` (full rewrite of the factory)
- Test: `tests/test_model.py` (new file)

**Interfaces:**
- Consumes: nothing new (no dependency on other tasks).
- Produces: `make_model(model: str | None = None, *, temperature: float = 0.0) -> ChatAnthropic` and `subagent_model() -> ChatAnthropic` — same names/signatures `nanoloop/agents.py` already imports and calls (`from .model import make_model, subagent_model`), so Task 1 alone keeps the rest of the package working.

- [ ] **Step 1: Swap the dependency in `pyproject.toml`**

In `pyproject.toml`, in the `dependencies` list, replace:

```toml
    "langchain-openai>=0.2.0",
```

with:

```toml
    "langchain-anthropic>=0.3.0",
```

- [ ] **Step 2: Install the new dependency**

Run: `pip install -e .`
Expected: installs `langchain-anthropic` and its transitive deps (including `anthropic` SDK); no errors.

- [ ] **Step 3: Write the failing test for missing API key**

Create `tests/test_model.py`:

```python
"""Anthropic model factory: env-driven config, missing-key error."""
from __future__ import annotations

import pytest

import nanoloop.model as model


def test_make_model_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        model.make_model()


def test_make_model_defaults_to_haiku(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("HARNESS_MODEL", raising=False)
    m = model.make_model()
    assert m.model == "claude-haiku-4-5"


def test_make_model_respects_harness_model_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("HARNESS_MODEL", "claude-sonnet-4-6")
    m = model.make_model()
    assert m.model == "claude-sonnet-4-6"


def test_subagent_model_defaults_to_haiku(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("HARNESS_SUBAGENT_MODEL", raising=False)
    m = model.subagent_model()
    assert m.model == "claude-haiku-4-5"
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError` or `AttributeError`, since `model.py` still builds `ChatOpenAI` against OpenRouter and raises on `OPENROUTER_API_KEY`, not `ANTHROPIC_API_KEY`.

- [ ] **Step 5: Rewrite `nanoloop/model.py`**

Replace the full file contents with:

```python
"""Anthropic model factory.

DeepAgents accepts any LangChain chat model object, so we build a
langchain_anthropic.ChatAnthropic pointed directly at the Anthropic API.
"""
from __future__ import annotations

import os

from langchain_anthropic import ChatAnthropic


def make_model(model: str | None = None, *, temperature: float = 0.0) -> ChatAnthropic:
    """Build a ChatAnthropic client.

    model: Anthropic model id, e.g. "claude-sonnet-4-6". Falls back to
    HARNESS_MODEL env var, then to "claude-haiku-4-5".
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set (see .env.example)")

    slug = model or os.environ.get("HARNESS_MODEL", "claude-haiku-4-5")

    # Cap output tokens; DeepAgents otherwise requests very large budgets.
    max_tokens = int(os.environ.get("HARNESS_MAX_TOKENS", "4096"))

    # Retry transient upstream errors (Anthropic 429/5xx).
    max_retries = int(os.environ.get("HARNESS_MAX_RETRIES", "5"))

    return ChatAnthropic(
        model=slug,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )


def subagent_model() -> ChatAnthropic:
    """Cheaper/faster model for role subagents; falls back to main model."""
    slug = os.environ.get("HARNESS_SUBAGENT_MODEL", "claude-haiku-4-5")
    return make_model(slug)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: PASS (4 passed).

- [ ] **Step 7: Run the full test suite to check for regressions**

Run: `pytest -v`
Expected: all tests pass (existing `test_memory.py`, `test_session.py`, `test_skills.py`, `test_tools.py` are untouched by this change and should be unaffected).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml nanoloop/model.py tests/test_model.py
git commit -m "feat: swap nanoLoop model layer from OpenRouter to Anthropic"
```

---

### Task 2: Update docs, comments, and `.env.example` to match

**Files:**
- Modify: `.env.example`
- Modify: `nanoloop/__init__.py:1`
- Modify: `nanoloop/agents.py:33`
- Modify: `README.md` (lines 8, 20, 44, 121 per current content)

**Interfaces:**
- Consumes: env var names and defaults fixed in Task 1 (`ANTHROPIC_API_KEY`, `HARNESS_MODEL`/`HARNESS_SUBAGENT_MODEL` defaulting to `claude-haiku-4-5`, `HARNESS_MAX_TOKENS`, `HARNESS_MAX_RETRIES`).
- Produces: nothing consumed by later tasks — this is the last task in the plan.

- [ ] **Step 1: Rewrite `.env.example`**

Replace the full file contents with:

```
# Anthropic — https://console.anthropic.com/settings/keys
ANTHROPIC_API_KEY=

# Main orchestrator model
HARNESS_MODEL=claude-haiku-4-5

# Subagent model (planner/builder/reviewer/qa/shipper)
HARNESS_SUBAGENT_MODEL=claude-haiku-4-5

# Cap output tokens per call
HARNESS_MAX_TOKENS=4096

# Retry transient Anthropic 429/5xx errors
HARNESS_MAX_RETRIES=5
```

- [ ] **Step 2: Update `nanoloop/__init__.py` docstring**

In `nanoloop/__init__.py`, replace line 1:

```python
"""nanoLoop: tiny autonomous engineering harness (OpenRouter + DeepAgents + OpenShell)."""
```

with:

```python
"""nanoLoop: tiny autonomous engineering harness (Anthropic + DeepAgents + OpenShell)."""
```

- [ ] **Step 3: Update `nanoloop/agents.py` docstring comment**

In `nanoloop/agents.py`, in the `build_agent` docstring, replace:

```python
    - Main model: OpenRouter (HARNESS_MODEL).
```

with:

```python
    - Main model: Anthropic API (HARNESS_MODEL).
```

- [ ] **Step 4: Update `README.md`**

Replace line 8:

```markdown
| Model | [OpenRouter](https://openrouter.ai) | One API, any tool-calling model. OpenAI-compatible → `ChatOpenAI` w/ custom `base_url`. |
```

with:

```markdown
| Model | [Anthropic API](https://console.anthropic.com) | Claude models directly via `ChatAnthropic`, no gateway layer. |
```

Replace line 20:

```markdown
             └─ DeepAgents orchestrator   [OpenRouter: HARNESS_MODEL]
```

with:

```markdown
             └─ DeepAgents orchestrator   [Anthropic: HARNESS_MODEL]
```

Replace line 44:

```markdown
cp .env.example .env        # add your OPENROUTER_API_KEY
```

with:

```markdown
cp .env.example .env        # add your ANTHROPIC_API_KEY
```

Replace line 121:

```markdown
- `nanoloop/model.py` — OpenRouter `ChatOpenAI` factory
```

with:

```markdown
- `nanoloop/model.py` — Anthropic `ChatAnthropic` factory
```

- [ ] **Step 5: Verify no OpenRouter references remain**

Run: `grep -rn "OpenRouter\|OPENROUTER\|ChatOpenAI\|openrouter" --include="*.py" --include="*.toml" --include="*.example" --include="*.md" . | grep -v docs/superpowers`
Expected: no output (empty).

- [ ] **Step 6: Run the full test suite one more time**

Run: `pytest -v`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add .env.example nanoloop/__init__.py nanoloop/agents.py README.md
git commit -m "docs: update nanoLoop docs/config for Anthropic model swap"
```
