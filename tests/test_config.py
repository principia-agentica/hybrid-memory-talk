import importlib
import pytest


def reload_config_with(monkeypatch, env: dict):
    # Apply env vars for this test
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    # Ensure clean reload
    if 'config' in list(importlib.sys.modules):
        importlib.reload(importlib.import_module('config'))
    else:
        import config  # noqa: F401
    # Final reload to pick up env
    import config
    return importlib.reload(config)


def test_config_defaults_without_env(monkeypatch):
    # Clear relevant env vars
    for k in [
        'HM_K_EPI','HM_K_SEM','HM_TOKEN_BUDGET','HM_EPISODIC_TTL_DAYS',
        'HM_EPI_FILTERS_JSON','HM_SEM_FILTERS_JSON','HM_RERANKER_ENABLED',
    ]:
        monkeypatch.delenv(k, raising=False)

    config = reload_config_with(monkeypatch, {})

    assert config.K_EPI == 4
    assert config.K_SEM == 3
    assert config.TOKEN_BUDGET == 1600
    assert config.EPISODIC_TTL_DAYS == 30
    # Defaults per module: episodic filters None; semantic filters default policy filter
    assert config.EPI_FILTERS is None
    assert config.SEM_FILTERS == {"tags": ["policy"], "pii": False}
    assert config.RERANKER_ENABLED is False


def test_config_env_overrides_and_json_parsing(monkeypatch):
    cfg = {
        'HM_K_EPI': '5',
        'HM_K_SEM': '4',
        'HM_TOKEN_BUDGET': '1201',
        'HM_EPISODIC_TTL_DAYS': '7',
        'HM_RERANKER_ENABLED': 'true',
        'HM_EPI_FILTERS_JSON': '{"session": "sess_1", "tags": ["password_reset"]}',
        'HM_SEM_FILTERS_JSON': '{"tags": ["policy", "prod"], "pii": false}',
    }
    config = reload_config_with(monkeypatch, cfg)

    assert config.K_EPI == 5
    assert config.K_SEM == 4
    assert config.TOKEN_BUDGET == 1201
    assert config.EPISODIC_TTL_DAYS == 7
    assert config.RERANKER_ENABLED is True
    assert config.EPI_FILTERS == {"session": "sess_1", "tags": ["password_reset"]}
    assert config.SEM_FILTERS == {"tags": ["policy", "prod"], "pii": False}

    # Invalid JSON should harmlessly fall back (here to previous default for SEM filters)
    monkeypatch.setenv('HM_SEM_FILTERS_JSON', '{not: valid json]')
    config = reload_config_with(monkeypatch, {})
    assert config.SEM_FILTERS == {"tags": ["policy"], "pii": False}


def test_components_pick_up_config_when_args_omitted(monkeypatch):
    # Override env and reload config first
    monkeypatch.setenv('HM_K_EPI', '2')
    monkeypatch.setenv('HM_K_SEM', '1')
    monkeypatch.setenv('HM_TOKEN_BUDGET', '999')
    monkeypatch.setenv('HM_RERANKER_ENABLED', '1')
    monkeypatch.setenv('HM_EPISODIC_TTL_DAYS', '42')
    # Episodic filter to require a tag that our event will have
    monkeypatch.setenv('HM_EPI_FILTERS_JSON', '{"tags": ["keep_me"]}')

    # Reload config and dependent modules
    import config as _config
    import memory.episodic_store as episodic_store
    import memory.semantic_store as semantic_store
    config = importlib.reload(_config)
    episodic_store = importlib.reload(episodic_store)

    # Build minimal stores
    epi = episodic_store.EpisodicStore()

    class TinyEncoder:
        def embed(self, text: str):
            return [1.0, 0.0]

    semantic_store = importlib.reload(semantic_store)
    sem = semantic_store.SemanticStore(encoder=TinyEncoder())
    sem.add({"id": "p1", "text": "policy about passwords", "metadata": {"tags": ["policy"]}})

    # Now reload hybrid_retriever after config is set
    import memory.hybrid_retriever as hybrid_retriever
    hybrid_retriever = importlib.reload(hybrid_retriever)

    retr = hybrid_retriever.HybridRetriever(episodic=epi, semantic=sem)

    # Check that retriever picked up config-derived defaults
    assert retr.k_epi == 2
    assert retr.k_sem == 1
    assert retr.token_budget == 999
    assert retr.reranker_enabled is True

    # EpisodicStore default_ttl_days should reflect config
    # (we can infer by checking the added event gets an expires_at ~42 days ahead)
    from datetime import datetime as _dt

    epi.add({"id": "e1", "text": "hello", "tags": ["keep_me"]})
    ev = list(epi)[-1]
    assert 'expires_at' in ev
    exp = ev['expires_at']
    ts = ev['ts']

    def parse_iso(s: str):
        s = s[:-1] if s.endswith('Z') else s
        try:
            return _dt.fromisoformat(s)
        except ValueError:
            return _dt.utcnow()

    exp_dt = parse_iso(exp)
    now = parse_iso(ts)
    assert 41 <= (exp_dt - now).days <= 43

    # Ensure the default episodic filter from config is active by retrieving
    items = retr.retrieve("password reset question")
    assert isinstance(items, list)
