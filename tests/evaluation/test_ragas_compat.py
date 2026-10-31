import sys

from project_rag.evaluation.ragas_compat import MISSING_MODULE, patch_vertexai_import


def test_patch_registers_stand_in_when_module_is_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, MISSING_MODULE, raising=False)
    monkeypatch.setitem(sys.modules, "langchain_community", None)
    patch_vertexai_import()
    assert hasattr(sys.modules[MISSING_MODULE], "ChatVertexAI")
