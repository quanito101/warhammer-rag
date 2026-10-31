from project_rag.llm.cache import ResponseCache, make_key


def test_key_is_stable_for_the_same_request():
    assert make_key("m", "sys", "hello") == make_key("m", "sys", "hello")


def test_key_changes_with_any_input():
    base = make_key("m", "sys", "hello")
    assert make_key("other", "sys", "hello") != base
    assert make_key("m", None, "hello") != base
    assert make_key("m", "sys", "hello!") != base


def test_missing_key_returns_none(tmp_path):
    assert ResponseCache(tmp_path / "cache.sqlite").get("absent") is None


def test_stored_response_is_returned(tmp_path):
    cache = ResponseCache(tmp_path / "cache.sqlite")
    cache.set("k", "answer")
    assert cache.get("k") == "answer"


def test_cache_persists_between_instances(tmp_path):
    path = tmp_path / "cache.sqlite"
    ResponseCache(path).set("k", "answer")
    assert ResponseCache(path).get("k") == "answer"


def test_set_overwrites_an_existing_key(tmp_path):
    cache = ResponseCache(tmp_path / "cache.sqlite")
    cache.set("k", "old")
    cache.set("k", "new")
    assert cache.get("k") == "new"
