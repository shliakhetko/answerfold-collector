from analysis.coding.cache import LLMCache, cache_key


def test_cache_key_is_stable_and_deterministic():
    key1 = cache_key(input_text="hello", model="claude-haiku-x", prompt_version="v1")
    key2 = cache_key(input_text="hello", model="claude-haiku-x", prompt_version="v1")
    assert key1 == key2
    assert len(key1) == 64  # sha256 hex digest


def test_cache_key_changes_with_any_component():
    base = cache_key(input_text="hello", model="m1", prompt_version="v1")
    assert base != cache_key(input_text="goodbye", model="m1", prompt_version="v1")
    assert base != cache_key(input_text="hello", model="m2", prompt_version="v1")
    assert base != cache_key(input_text="hello", model="m1", prompt_version="v2")


def test_cache_key_no_separator_collision():
    # Without a separator, ("ab", "c") and ("a", "bc") would collide when
    # naively concatenated -- the NUL-byte join must prevent this.
    a = cache_key(input_text="c", model="ab", prompt_version="v1")
    b = cache_key(input_text="bc", model="a", prompt_version="v1")
    assert a != b


def test_get_returns_none_when_absent(tmp_path):
    cache = LLMCache(cache_dir=tmp_path / "extractor")
    assert cache.get("does-not-exist") is None


def test_set_then_get_round_trips(tmp_path):
    cache = LLMCache(cache_dir=tmp_path / "extractor")
    key = cache_key(input_text="hello", model="m1", prompt_version="v1")
    value = {"mentions": [{"surface_form": "Smith.ai"}], "tokens_input": 120, "tokens_output": 40}
    cache.set(key, value)
    assert cache.get(key) == value


def test_set_creates_cache_dir(tmp_path):
    cache_dir = tmp_path / "nested" / "extractor"
    assert not cache_dir.exists()
    cache = LLMCache(cache_dir=cache_dir)
    cache.set("k", {"a": 1})
    assert cache_dir.exists()


def test_different_keys_do_not_collide(tmp_path):
    cache = LLMCache(cache_dir=tmp_path / "extractor")
    cache.set("key-a", {"v": "a"})
    cache.set("key-b", {"v": "b"})
    assert cache.get("key-a") == {"v": "a"}
    assert cache.get("key-b") == {"v": "b"}
