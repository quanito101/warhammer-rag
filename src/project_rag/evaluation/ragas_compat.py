"""Makes `import ragas` work with the langchain-community version this
project uses.

ragas 0.4.3 imports langchain_community.chat_models.vertexai.ChatVertexAI at
load time, only to list it among model classes. langchain-community 0.4
removed that module, and the older release that has it requires
langchain-core below 1.0. This registers an empty stand-in when the real
module is missing; nothing in this project calls Vertex AI.
"""

import sys
import types

MISSING_MODULE = "langchain_community.chat_models.vertexai"


def patch_vertexai_import() -> None:
    """Registers a stand-in ChatVertexAI module if the real one is absent."""
    try:
        __import__(MISSING_MODULE)
    except ImportError:
        stand_in = types.ModuleType(MISSING_MODULE)
        stand_in.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[MISSING_MODULE] = stand_in
