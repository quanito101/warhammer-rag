"""Where the Gemini API key comes from.

The key is looked up in this order: the GEMINI_API_KEY environment variable,
a `.env` file at the project root, and, in the interactive chat only, a
hidden prompt. Whatever is found is placed in the environment, so the
Gemini client and the LangChain model both see it. The key is never written
to disk by this module.
"""

import getpass
import os
from pathlib import Path

API_KEY_VARIABLE = "GEMINI_API_KEY"
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
MISSING_KEY_MESSAGE = (
    f"{API_KEY_VARIABLE} is not set. Create a key in Google AI Studio, then either "
    f"export {API_KEY_VARIABLE}=<key>, or put {API_KEY_VARIABLE}=<key> in a file named .env "
    "at the project root (see .env.example)."
)


def clean_key(raw: str) -> str:
    """Removes whitespace, quotes and non-printable characters that a copy
    and paste can add around a key and that make the API reject it.
    """
    return "".join(char for char in raw if char.isprintable()).strip().strip("'\"").strip()


def parse_env_file(text: str) -> dict[str, str]:
    """KEY=value pairs from the text of a .env file. Blank lines and lines
    starting with # are ignored, and a leading `export ` is accepted.
    """
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        values[name.removeprefix("export ").strip()] = clean_key(value)
    return values


def ensure_api_key(prompt_if_missing: bool = False, env_file: Path = ENV_FILE) -> str:
    """Returns the API key and makes sure it is in the environment. Raises a
    RuntimeError that explains how to provide the key when none is found
    and prompting is not allowed.
    """
    key = clean_key(os.environ.get(API_KEY_VARIABLE, ""))
    if not key and env_file.exists():
        key = parse_env_file(env_file.read_text()).get(API_KEY_VARIABLE, "")
    if not key and prompt_if_missing:
        key = clean_key(getpass.getpass(f"{API_KEY_VARIABLE} (input is hidden): "))
    if not key:
        raise RuntimeError(MISSING_KEY_MESSAGE)
    os.environ[API_KEY_VARIABLE] = key
    return key
