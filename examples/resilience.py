from modelcore.application import ResilientProvider, RetryPolicy


def with_resilience(provider):
    return ResilientProvider(provider, RetryPolicy(max_attempts=3, base_delay=0.5), timeout=30)
