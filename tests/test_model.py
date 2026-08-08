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
