from modelcore.application import CheapPolicy, ModelCandidate, RoutingProvider


def router(provider):
    candidate = ModelCandidate("local", provider, "your-model", cost_score=1, latency_score=1, quality_score=1)
    return RoutingProvider(CheapPolicy(), [candidate])
