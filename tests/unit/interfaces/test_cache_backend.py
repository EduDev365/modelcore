from typing import get_type_hints

from modelcore.interfaces.cache_backend import CacheBackend
from modelcore.models.chat_response import ChatResponse


def test_cache_backend_has_only_get_and_set_contract() -> None:
    assert set(CacheBackend.__dict__) >= {"get", "set"}
    assert "delete" not in CacheBackend.__dict__
    assert "clear" not in CacheBackend.__dict__

    get_hints = get_type_hints(CacheBackend.get)
    set_hints = get_type_hints(CacheBackend.set)

    assert get_hints["key"] is str
    assert get_hints["return"] == ChatResponse | None
    assert set_hints["key"] is str
    assert set_hints["value"] is ChatResponse
    assert set_hints["ttl"] == float | None
