import pytest

from project_rag.config import API_KEY_VARIABLE, clean_key, ensure_api_key, parse_env_file


def test_clean_key_removes_whitespace_quotes_and_control_characters():
    assert clean_key('  "abc123"\r\n') == "abc123"
    assert clean_key("abc\x1b[200~123") == "abc[200~123"


def test_parse_env_file_ignores_comments_and_accepts_export():
    text = "# comment\n\nexport GEMINI_API_KEY='k1'\nOTHER=2\nnot a pair\n"
    assert parse_env_file(text) == {"GEMINI_API_KEY": "k1", "OTHER": "2"}


def test_environment_variable_wins_over_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=from-file\n")
    monkeypatch.setenv(API_KEY_VARIABLE, "from-env")
    assert ensure_api_key(env_file=env_file) == "from-env"


def test_env_file_is_used_and_exported(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=from-file\n")
    monkeypatch.delenv(API_KEY_VARIABLE, raising=False)
    assert ensure_api_key(env_file=env_file) == "from-file"
    import os

    assert os.environ[API_KEY_VARIABLE] == "from-file"


def test_prompt_is_used_only_when_allowed(monkeypatch, tmp_path):
    monkeypatch.delenv(API_KEY_VARIABLE, raising=False)
    monkeypatch.setattr("getpass.getpass", lambda prompt: " typed-key \n")
    missing = tmp_path / "absent.env"
    assert ensure_api_key(prompt_if_missing=True, env_file=missing) == "typed-key"
    monkeypatch.delenv(API_KEY_VARIABLE)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        ensure_api_key(env_file=missing)
