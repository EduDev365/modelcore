from modelcore.application import FallbackProvider


def with_fallback(primary, secondary):
    return FallbackProvider([primary, secondary])
