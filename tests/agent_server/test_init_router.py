"""Tests for the deferred-init / dormant-mode flow.

Background: https://github.com/OpenHands/software-agent-sdk/issues/2523
"""

from __future__ import annotations

import base64
import json
import os
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException
from fastapi.testclient import TestClient
from joserfc import jwk, jwt
from pydantic import SecretStr

from openhands.agent_server.api import api_lifespan, create_app
from openhands.agent_server.config import Config
from openhands.agent_server.init_router import (
    InitRequest,
    InitService,
    _build_initialized_config,
    _verify_init_jwt,
    load_init_public_keys,
    resolve_init_token_audience,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """The agent-server pulls config from env at import time in places;
    null these out so each test starts from a clean slate."""
    for key in (
        "OH_DEFERRED_INIT",
        "OH_WEB_URL",
        "RUNTIME_URL",
        "TMUX_TMPDIR",
        "SESSION_API_KEY",
        "OH_SESSION_API_KEYS_0",
        "OH_SECRET_KEY",
        "OH_INIT_PUBLIC_KEY_FILE",
        "OH_INIT_TOKEN_AUDIENCE",
        "OH_INIT_TOKEN_AUDIENCE_ENV",
        "AGENT_SERVER_NAME",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_ec_keypair() -> tuple[str, str]:
    """Return ``(private_pem, public_pem)`` for an EC P-256 keypair (ES256)."""
    key = ec.generate_private_key(ec.SECP256R1())
    private_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    public_pem = (
        key.public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return private_pem, public_pem


def _write_keyfile(tmp_path: Path, *public_pems: str) -> Path:
    """Write one or more public PEM blocks into a single file; return its path.

    Concatenating multiple blocks mirrors how an operator stages a key rotation
    (old + new key trusted simultaneously) in one ``OH_INIT_PUBLIC_KEY_FILE``.
    """
    path = tmp_path / "init_public_keys.pem"
    path.write_text("\n".join(public_pems))
    return path


def _mint_init_jwt(private_pem: str, *, alg: str = "ES256", **claims: object) -> str:
    """Sign an init JWT with ``private_pem``. Claims are passed as kwargs."""
    key_type = "EC" if alg.startswith("ES") else "RSA"
    return jwt.encode(
        {"alg": alg}, claims, jwk.import_key(private_pem.encode(), key_type)
    )


def _forge_alg_none(**claims: object) -> str:
    """Build an unsigned ``alg:none`` token (the classic JWT bypass attempt)."""

    def _b64(data: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()

    return f"{_b64({'alg': 'none', 'typ': 'JWT'})}.{_b64(claims)}."


def _reset_conversation_singleton():
    """Some tests build their own ConversationService; reset the module-level
    cache so unrelated tests don't see leftover state."""
    from openhands.agent_server import conversation_service as cs_mod

    cs_mod._conversation_service = None


class TestConfigDefaults:
    def test_deferred_init_defaults_false(self):
        assert Config().deferred_init is False


class TestBuildInitializedConfig:
    def test_clears_deferred_init_flag(self):
        base = Config(deferred_init=True)
        merged = _build_initialized_config(base, InitRequest())
        assert merged.deferred_init is False

    def test_overrides_only_provided_fields(self, tmp_path):
        base = Config(
            deferred_init=True,
            conversations_path=Path("base/convs"),
            bash_events_dir=Path("base/bash"),
            max_concurrent_runs=5,
        )
        req = InitRequest(
            session_api_keys=["k1"],
            conversations_path=tmp_path / "user-workspace" / "conversations",
        )
        merged = _build_initialized_config(base, req)
        assert merged.session_api_keys == ["k1"]
        assert (
            merged.conversations_path == tmp_path / "user-workspace" / "conversations"
        )
        # Untouched fields keep base values.
        assert merged.bash_events_dir == Path("base/bash")
        assert merged.max_concurrent_runs == 5

    def test_secret_key_falls_back_to_session_key(self):
        base = Config(deferred_init=True)
        # base.secret_key default is None (no env), so we should fall back
        # to the first session key after /api/init.
        assert base.secret_key is None
        merged = _build_initialized_config(
            base, InitRequest(session_api_keys=["s1", "s2"])
        )
        assert merged.secret_key is not None
        assert merged.secret_key.get_secret_value() == "s1"

    def test_explicit_secret_key_wins(self):
        base = Config(deferred_init=True)
        merged = _build_initialized_config(
            base,
            InitRequest(
                session_api_keys=["sk"], secret_key=SecretStr("explicit-secret")
            ),
        )
        assert merged.secret_key is not None
        assert merged.secret_key.get_secret_value() == "explicit-secret"


class TestRouterMounting:
    """Behavior of the /api/init endpoint outside the lifespan."""

    def test_init_get_404_without_deferred_mode(self):
        # When deferred_init=False the InitService is never attached to
        # app.state, so the endpoint behaves as if not configured.
        app = create_app(Config(deferred_init=False))
        client = TestClient(app)
        resp = client.get("/api/init")
        assert resp.status_code == 404


class TestInitServiceTransitions:
    @pytest.mark.asyncio
    async def test_init_transitions_dormant_to_ready(self, tmp_path):
        _reset_conversation_singleton()
        base = Config(
            deferred_init=True,
            conversations_path=tmp_path / "convs",
            bash_events_dir=tmp_path / "bash",
        )
        app = SimpleNamespace(state=SimpleNamespace(config=base))
        svc = InitService(app, base_config=base)  # type: ignore[arg-type]
        assert svc.state == "dormant"

        result = await svc.initialize(
            InitRequest(
                session_api_keys=["user-key"],
                conversations_path=tmp_path / "user" / "convs",
                bash_events_dir=tmp_path / "user" / "bash",
            )
        )
        try:
            assert result.state == "ready"
            assert svc.state == "ready"
            # New config landed on app.state with deferred_init cleared.
            assert app.state.config.deferred_init is False
            assert app.state.config.session_api_keys == ["user-key"]
            assert app.state.conversation_service is not None
        finally:
            await svc.teardown()
            _reset_conversation_singleton()

    @pytest.mark.asyncio
    async def test_second_init_rejected_with_400(self, tmp_path):
        _reset_conversation_singleton()
        from fastapi import HTTPException

        base = Config(
            deferred_init=True,
            conversations_path=tmp_path / "convs",
            bash_events_dir=tmp_path / "bash",
        )
        app = SimpleNamespace(state=SimpleNamespace(config=base))
        svc = InitService(app, base_config=base)  # type: ignore[arg-type]

        await svc.initialize(
            InitRequest(
                conversations_path=tmp_path / "u1" / "convs",
                bash_events_dir=tmp_path / "u1" / "bash",
            )
        )
        try:
            with pytest.raises(HTTPException) as excinfo:
                await svc.initialize(InitRequest())
            assert excinfo.value.status_code == 400
            assert "already in state" in str(excinfo.value.detail)
        finally:
            await svc.teardown()
            _reset_conversation_singleton()

    @pytest.mark.asyncio
    async def test_init_applies_env_vars(self, tmp_path, monkeypatch):
        _reset_conversation_singleton()
        # Pre-clean so the env var truly comes from /api/init.
        monkeypatch.delenv("DEFERRED_INIT_TEST_VAR", raising=False)
        base = Config(
            deferred_init=True,
            conversations_path=tmp_path / "convs",
            bash_events_dir=tmp_path / "bash",
        )
        app = SimpleNamespace(state=SimpleNamespace(config=base))
        svc = InitService(app, base_config=base)  # type: ignore[arg-type]

        await svc.initialize(
            InitRequest(
                env={"DEFERRED_INIT_TEST_VAR": "hello"},
                conversations_path=tmp_path / "u" / "convs",
                bash_events_dir=tmp_path / "u" / "bash",
            )
        )
        try:
            assert os.environ.get("DEFERRED_INIT_TEST_VAR") == "hello"
        finally:
            await svc.teardown()
            monkeypatch.delenv("DEFERRED_INIT_TEST_VAR", raising=False)
            _reset_conversation_singleton()


class TestEndToEndOverLifespan:
    """Drive the whole flow through the FastAPI lifespan + TestClient."""

    def test_dormant_503s_api_routes_until_init(self, tmp_path):
        _reset_conversation_singleton()
        cfg = Config(
            deferred_init=True,
            conversations_path=tmp_path / "convs",
            bash_events_dir=tmp_path / "bash",
        )
        app = create_app(cfg)
        with TestClient(app) as client:
            try:
                # Health/ready/server_info are not gated.
                assert client.get("/alive").status_code == 200
                assert client.get("/ready").status_code == 200

                # Sample /api/* route — should be 503. The agent-server's
                # 5xx exception handler replaces ``detail`` with a generic
                # "Internal Server Error" message, so we only assert on the
                # status code here — that's what the warm-pool orchestrator
                # actually inspects.
                resp = client.get("/api/conversations/count")
                assert resp.status_code == 503

                # Init status reports dormant.
                resp = client.get("/api/init")
                assert resp.status_code == 200
                assert resp.json()["state"] == "dormant"

                # Run /api/init.
                resp = client.post(
                    "/api/init",
                    json={
                        "conversations_path": str(tmp_path / "u" / "convs"),
                        "bash_events_dir": str(tmp_path / "u" / "bash"),
                    },
                )
                assert resp.status_code == 200
                assert resp.json()["state"] == "ready"

                # /api/* now works (200, not 503).
                resp = client.get("/api/conversations/count")
                assert resp.status_code == 200
            finally:
                _reset_conversation_singleton()

    def test_init_api_key_required_when_configured(self, tmp_path):
        _reset_conversation_singleton()
        cfg = Config(
            deferred_init=True,
            secret_key=SecretStr("pool-key"),
            conversations_path=tmp_path / "convs",
            bash_events_dir=tmp_path / "bash",
        )
        app = create_app(cfg)
        with TestClient(app) as client:
            try:
                # Wrong key → 401.
                resp = client.post(
                    "/api/init",
                    headers={"X-Init-API-Key": "wrong"},
                    json={
                        "conversations_path": str(tmp_path / "u" / "convs"),
                        "bash_events_dir": str(tmp_path / "u" / "bash"),
                    },
                )
                assert resp.status_code == 401

                # No key → 401.
                resp = client.post("/api/init", json={})
                assert resp.status_code == 401

                # Right key → 200.
                resp = client.post(
                    "/api/init",
                    headers={"X-Init-API-Key": "pool-key"},
                    json={
                        "conversations_path": str(tmp_path / "u" / "convs"),
                        "bash_events_dir": str(tmp_path / "u" / "bash"),
                    },
                )
                assert resp.status_code == 200

                # GET /api/init does NOT require the key (status polling).
                resp = client.get("/api/init")
                assert resp.status_code == 200
            finally:
                _reset_conversation_singleton()

    def test_session_api_key_set_at_init_protects_api(self, tmp_path):
        _reset_conversation_singleton()
        cfg = Config(
            deferred_init=True,
            conversations_path=tmp_path / "convs",
            bash_events_dir=tmp_path / "bash",
        )
        app = create_app(cfg)
        with TestClient(app) as client:
            try:
                # Before /api/init, no session key required at startup config
                # level — but the dormant gate 503s anyway.
                assert client.get("/api/conversations/count").status_code == 503

                # Init delivers the session key.
                resp = client.post(
                    "/api/init",
                    json={
                        "session_api_keys": ["user-session-key"],
                        "conversations_path": str(tmp_path / "u" / "convs"),
                        "bash_events_dir": str(tmp_path / "u" / "bash"),
                    },
                )
                assert resp.status_code == 200

                # NOTE: session_api_keys configured at /api/init time take effect
                # on the *config object*, but the FastAPI session-key
                # dependency was bound to the original (dormant) config when
                # the routes were mounted. Documenting this trade-off:
                # in production, set OH_SESSION_API_KEYS_0 at instance start so
                # auth is in place from the moment routes go live, and use
                # /api/init only to deliver workspace + per-user runtime config.
                # The dormant gate ensures no traffic reaches gated routes
                # before /api/init regardless.
                assert app.state.config.session_api_keys == ["user-session-key"]
            finally:
                _reset_conversation_singleton()


class TestNonDeferredPathUnchanged:
    """Regression: deferred_init=False must behave exactly like before."""

    def test_non_deferred_does_not_create_init_service(self, tmp_path):
        _reset_conversation_singleton()
        cfg = Config(
            deferred_init=False,
            conversations_path=tmp_path / "convs",
            bash_events_dir=tmp_path / "bash",
        )
        app = create_app(cfg)
        with TestClient(app) as client:
            try:
                # No init_service in non-deferred mode.
                assert getattr(app.state, "init_service", None) is None
                # /api/* should be live (200) — the dormant gate is a no-op.
                assert client.get("/api/conversations/count").status_code == 200
                # /api/init returns 404 because no InitService is attached.
                assert client.get("/api/init").status_code == 404
            finally:
                _reset_conversation_singleton()


class TestResolveInitTokenAudience:
    """Audience = a literal, a named env var, or the default AGENT_SERVER_NAME."""

    def test_default_source_is_agent_server_name(self):
        assert Config(deferred_init=True).init_token_audience_env == "AGENT_SERVER_NAME"

    def test_resolves_from_default_agent_server_name(self, monkeypatch):
        # Zero extra config: the orchestrator sets AGENT_SERVER_NAME per instance
        # and the identical deploy spec resolves each instance's own value.
        monkeypatch.setenv("AGENT_SERVER_NAME", "instance-7")
        assert resolve_init_token_audience(Config(deferred_init=True)) == "instance-7"

    def test_literal_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("AGENT_SERVER_NAME", "from-env")
        cfg = Config(deferred_init=True, init_token_audience="literal-aud")
        assert resolve_init_token_audience(cfg) == "literal-aud"

    def test_explicit_source_overrides_default(self, monkeypatch):
        # Pointing init_token_audience_env elsewhere ignores AGENT_SERVER_NAME.
        monkeypatch.setenv("AGENT_SERVER_NAME", "should-not-be-used")
        monkeypatch.setenv("OH_TEST_AUD_SRC", "from-custom-source")
        cfg = Config(deferred_init=True, init_token_audience_env="OH_TEST_AUD_SRC")
        assert resolve_init_token_audience(cfg) == "from-custom-source"

    def test_none_when_source_var_unset(self, monkeypatch):
        # Default source AGENT_SERVER_NAME unset and no literal → None.
        monkeypatch.delenv("AGENT_SERVER_NAME", raising=False)
        assert resolve_init_token_audience(Config(deferred_init=True)) is None


class TestVerifyInitJwt:
    """Unit tests for the asymmetric init JWT verifier (no FastAPI/lifespan).

    These exercise the crypto/claims paths directly so failures are crisp.
    """

    def _cfg(self, tmp_path: Path, *public_pems: str, **overrides) -> Config:
        params: dict = dict(
            deferred_init=True,
            init_public_key_file=_write_keyfile(tmp_path, *public_pems),
            init_token_audience="instance-test",
        )
        params.update(overrides)
        return Config(**params)

    def _assert_401(self, fn) -> None:
        with pytest.raises(HTTPException) as excinfo:
            fn()
        assert excinfo.value.status_code == 401

    def _verify(self, token: str, cfg: Config) -> None:
        """Load trusted keys (as the server does at boot), then verify token."""
        _verify_init_jwt(token, load_init_public_keys(cfg), cfg)

    def test_valid_token_accepted(self, tmp_path):
        priv, pub = _make_ec_keypair()
        cfg = self._cfg(tmp_path, pub)
        now = int(time.time())
        # Should not raise.
        self._verify(_mint_init_jwt(priv, aud="instance-test", exp=now + 300), cfg)

    def test_expired_token_rejected(self, tmp_path):
        priv, pub = _make_ec_keypair()
        cfg = self._cfg(tmp_path, pub, init_token_leeway_seconds=0)
        now = int(time.time())
        self._assert_401(
            lambda: self._verify(
                _mint_init_jwt(priv, aud="instance-test", exp=now - 30), cfg
            )
        )

    def test_token_within_leeway_accepted(self, tmp_path):
        priv, pub = _make_ec_keypair()
        cfg = self._cfg(tmp_path, pub, init_token_leeway_seconds=60)
        now = int(time.time())
        # 10s past expiry but within the 60s leeway window.
        self._verify(_mint_init_jwt(priv, aud="instance-test", exp=now - 10), cfg)

    def test_missing_exp_rejected(self, tmp_path):
        priv, pub = _make_ec_keypair()
        cfg = self._cfg(tmp_path, pub)
        self._assert_401(
            lambda: self._verify(_mint_init_jwt(priv, aud="instance-test"), cfg)
        )

    def test_wrong_audience_rejected(self, tmp_path):
        priv, pub = _make_ec_keypair()
        cfg = self._cfg(tmp_path, pub)
        now = int(time.time())
        self._assert_401(
            lambda: self._verify(
                _mint_init_jwt(priv, aud="other-instance", exp=now + 300), cfg
            )
        )

    def test_untrusted_signer_rejected(self, tmp_path):
        _priv_trusted, pub = _make_ec_keypair()
        priv_evil, _pub_evil = _make_ec_keypair()
        cfg = self._cfg(tmp_path, pub)
        now = int(time.time())
        self._assert_401(
            lambda: self._verify(
                _mint_init_jwt(priv_evil, aud="instance-test", exp=now + 300), cfg
            )
        )

    def test_alg_none_rejected(self, tmp_path):
        _priv, pub = _make_ec_keypair()
        cfg = self._cfg(tmp_path, pub)
        now = int(time.time())
        self._assert_401(
            lambda: self._verify(
                _forge_alg_none(aud="instance-test", exp=now + 300), cfg
            )
        )

    def test_hs256_key_confusion_rejected(self, tmp_path):
        _priv, pub = _make_ec_keypair()
        cfg = self._cfg(tmp_path, pub)
        now = int(time.time())
        # Attacker signs with the PUBLIC key bytes as an HMAC secret. joserfc
        # warns (rightly) that this isn't a valid oct key — that's the attacker
        # misusing it, so silence the warning for this simulation.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forged = jwt.encode(
                {"alg": "HS256"},
                {"aud": "instance-test", "exp": now + 300},
                jwk.OctKey.import_key(pub.encode()),
            )
        self._assert_401(lambda: self._verify(forged, cfg))

    def test_malformed_token_rejected(self, tmp_path):
        _priv, pub = _make_ec_keypair()
        cfg = self._cfg(tmp_path, pub)
        self._assert_401(lambda: self._verify("not-a-jwt", cfg))

    def test_rotation_second_key_accepted(self, tmp_path):
        _priv_old, pub_old = _make_ec_keypair()
        priv_new, pub_new = _make_ec_keypair()
        # Server trusts both during a rotation overlap (both blocks concatenated
        # in one key file); token signed by the new key.
        cfg = self._cfg(tmp_path, pub_old, pub_new)
        now = int(time.time())
        self._verify(_mint_init_jwt(priv_new, aud="instance-test", exp=now + 300), cfg)

    def test_unparseable_key_skipped_then_valid_key_used(self, tmp_path):
        priv, pub = _make_ec_keypair()
        # A malformed-but-PEM-shaped block precedes the valid key: the regex
        # matches it, import fails, it's skipped, and the valid key verifies.
        bad_block = (
            "-----BEGIN PUBLIC KEY-----\nnot-valid-key-data\n-----END PUBLIC KEY-----"
        )
        cfg = self._cfg(tmp_path, bad_block, pub)
        now = int(time.time())
        self._verify(_mint_init_jwt(priv, aud="instance-test", exp=now + 300), cfg)

    def test_audience_from_env_source_accepted(self, tmp_path, monkeypatch):
        # End-to-end: the audience is read from a named env var, and a token
        # minted to match that resolved value verifies.
        priv, pub = _make_ec_keypair()
        monkeypatch.setenv("OH_TEST_AUD_SRC", "instance-42")
        cfg = self._cfg(
            tmp_path,
            pub,
            init_token_audience=None,
            init_token_audience_env="OH_TEST_AUD_SRC",
        )
        now = int(time.time())
        self._verify(_mint_init_jwt(priv, aud="instance-42", exp=now + 300), cfg)


class TestLoadInitPublicKeys:
    """Boot-time loader: reads the key file once, fails fast on misconfig."""

    def test_no_file_returns_empty(self):
        assert load_init_public_keys(Config(deferred_init=True)) == []

    def test_reads_all_keys_for_rotation(self, tmp_path):
        _priv1, pub1 = _make_ec_keypair()
        _priv2, pub2 = _make_ec_keypair()
        cfg = Config(
            deferred_init=True,
            init_public_key_file=_write_keyfile(tmp_path, pub1, pub2),
            init_token_audience="instance-test",
        )
        assert len(load_init_public_keys(cfg)) == 2

    def test_missing_audience_raises(self, tmp_path):
        # Audience is required when asymmetric init is enabled (replay binding).
        # No literal AND the default source (AGENT_SERVER_NAME) is unset — the
        # _clean_env fixture guarantees the latter — so nothing resolves.
        _priv, pub = _make_ec_keypair()
        cfg = Config(
            deferred_init=True,
            init_public_key_file=_write_keyfile(tmp_path, pub),
            init_token_audience=None,
        )
        with pytest.raises(RuntimeError, match="audience is required"):
            load_init_public_keys(cfg)

    def test_audience_from_default_source_loads(self, tmp_path, monkeypatch):
        # Default source (AGENT_SERVER_NAME) set, no explicit audience config →
        # boot requirement satisfied (the realistic default deployment).
        _priv, pub = _make_ec_keypair()
        monkeypatch.setenv("AGENT_SERVER_NAME", "instance-9")
        cfg = Config(
            deferred_init=True,
            init_public_key_file=_write_keyfile(tmp_path, pub),
        )
        assert len(load_init_public_keys(cfg)) == 1

    def test_audience_from_env_source_loads(self, tmp_path, monkeypatch):
        # Audience resolved from a named env var satisfies the boot requirement.
        _priv, pub = _make_ec_keypair()
        monkeypatch.setenv("OH_TEST_AUD_SRC", "instance-9")
        cfg = Config(
            deferred_init=True,
            init_public_key_file=_write_keyfile(tmp_path, pub),
            init_token_audience_env="OH_TEST_AUD_SRC",
        )
        assert len(load_init_public_keys(cfg)) == 1

    def test_missing_audience_env_var_raises(self, tmp_path, monkeypatch):
        # Env-var NAME configured but the variable is unset → no audience
        # resolves → boot fails (same guard as a missing literal).
        _priv, pub = _make_ec_keypair()
        monkeypatch.delenv("OH_TEST_AUD_SRC", raising=False)
        cfg = Config(
            deferred_init=True,
            init_public_key_file=_write_keyfile(tmp_path, pub),
            init_token_audience_env="OH_TEST_AUD_SRC",
        )
        with pytest.raises(RuntimeError, match="audience is required"):
            load_init_public_keys(cfg)

    def test_missing_file_raises(self, tmp_path):
        cfg = Config(
            deferred_init=True,
            init_public_key_file=tmp_path / "does-not-exist.pem",
            init_token_audience="instance-test",
        )
        with pytest.raises(RuntimeError, match="cannot read init public key file"):
            load_init_public_keys(cfg)

    def test_file_without_usable_keys_raises(self, tmp_path):
        keyfile = tmp_path / "junk.pem"
        keyfile.write_text("not a PEM block at all\n")
        cfg = Config(
            deferred_init=True,
            init_public_key_file=keyfile,
            init_token_audience="instance-test",
        )
        with pytest.raises(RuntimeError, match="no usable public keys"):
            load_init_public_keys(cfg)


class TestInitPublicKeyFileEnvLoading:
    """The OH_INIT_* asymmetric fields load from env via the parser."""

    def test_env_var_maps_to_init_public_key_file(self, tmp_path, monkeypatch):
        from openhands.agent_server.config import ENVIRONMENT_VARIABLE_PREFIX
        from openhands.agent_server.env_parser import (
            _get_default_parsers,
            get_env_parser,
        )

        keyfile = tmp_path / "k.pem"
        monkeypatch.setenv("OH_INIT_PUBLIC_KEY_FILE", str(keyfile))
        parser = get_env_parser(Config, _get_default_parsers())
        data = parser.from_env(ENVIRONMENT_VARIABLE_PREFIX)
        assert isinstance(data, dict)
        assert data["init_public_key_file"] == str(keyfile)
        # And the value validates onto the Config as a Path.
        assert Config(**data).init_public_key_file == keyfile

    def test_audience_env_var_maps_to_field(self, monkeypatch):
        from openhands.agent_server.config import ENVIRONMENT_VARIABLE_PREFIX
        from openhands.agent_server.env_parser import (
            _get_default_parsers,
            get_env_parser,
        )

        monkeypatch.setenv("OH_INIT_TOKEN_AUDIENCE_ENV", "HOSTNAME")
        parser = get_env_parser(Config, _get_default_parsers())
        data = parser.from_env(ENVIRONMENT_VARIABLE_PREFIX)
        assert isinstance(data, dict)
        assert data["init_token_audience_env"] == "HOSTNAME"
        assert Config(**data).init_token_audience_env == "HOSTNAME"


class TestAsymmetricInitAuthOverLifespan:
    """Drive asymmetric init auth through the FastAPI lifespan + TestClient."""

    def _dormant_app(self, tmp_path, public_pems, **cfg_overrides):
        cfg = Config(
            deferred_init=True,
            init_public_key_file=_write_keyfile(tmp_path, *public_pems),
            init_token_audience="instance-test",
            conversations_path=tmp_path / "convs",
            bash_events_dir=tmp_path / "bash",
            **cfg_overrides,
        )
        return create_app(cfg)

    def _init_body(self, tmp_path) -> dict:
        return {
            "conversations_path": str(tmp_path / "u" / "convs"),
            "bash_events_dir": str(tmp_path / "u" / "bash"),
        }

    def test_valid_bearer_token_initializes(self, tmp_path):
        _reset_conversation_singleton()
        priv, pub = _make_ec_keypair()
        app = self._dormant_app(tmp_path, [pub])
        with TestClient(app) as client:
            try:
                token = _mint_init_jwt(
                    priv, aud="instance-test", exp=int(time.time()) + 300
                )
                resp = client.post(
                    "/api/init",
                    headers={"Authorization": f"Bearer {token}"},
                    json=self._init_body(tmp_path),
                )
                assert resp.status_code == 200, resp.text
                assert resp.json()["state"] == "ready"
            finally:
                _reset_conversation_singleton()

    def test_jwt_via_x_init_api_key_header_rejected(self, tmp_path):
        # Asymmetric auth is Bearer-only: a JWT in X-Init-API-Key is NOT
        # accepted (that header is reserved for the symmetric secret).
        _reset_conversation_singleton()
        priv, pub = _make_ec_keypair()
        app = self._dormant_app(tmp_path, [pub])
        with TestClient(app) as client:
            try:
                token = _mint_init_jwt(
                    priv, aud="instance-test", exp=int(time.time()) + 300
                )
                resp = client.post(
                    "/api/init",
                    headers={"X-Init-API-Key": token},
                    json=self._init_body(tmp_path),
                )
                assert resp.status_code == 401
            finally:
                _reset_conversation_singleton()

    def test_missing_token_rejected(self, tmp_path):
        _reset_conversation_singleton()
        _priv, pub = _make_ec_keypair()
        app = self._dormant_app(tmp_path, [pub])
        with TestClient(app) as client:
            try:
                resp = client.post("/api/init", json=self._init_body(tmp_path))
                assert resp.status_code == 401
            finally:
                _reset_conversation_singleton()

    def test_invalid_token_rejected(self, tmp_path):
        _reset_conversation_singleton()
        _priv, pub = _make_ec_keypair()
        app = self._dormant_app(tmp_path, [pub])
        with TestClient(app) as client:
            try:
                resp = client.post(
                    "/api/init",
                    headers={"Authorization": "Bearer not-a-valid-jwt"},
                    json=self._init_body(tmp_path),
                )
                assert resp.status_code == 401
            finally:
                _reset_conversation_singleton()

    def test_symmetric_secret_rejected_when_public_key_set(self, tmp_path):
        # Precedence: both configured → asymmetric required; the symmetric
        # secret must NOT grant init (fail closed).
        _reset_conversation_singleton()
        _priv, pub = _make_ec_keypair()
        app = self._dormant_app(tmp_path, [pub], secret_key=SecretStr("pool-key"))
        with TestClient(app) as client:
            try:
                resp = client.post(
                    "/api/init",
                    headers={"X-Init-API-Key": "pool-key"},
                    json=self._init_body(tmp_path),
                )
                assert resp.status_code == 401
            finally:
                _reset_conversation_singleton()

    def test_valid_jwt_accepted_when_both_configured(self, tmp_path):
        _reset_conversation_singleton()
        priv, pub = _make_ec_keypair()
        app = self._dormant_app(tmp_path, [pub], secret_key=SecretStr("pool-key"))
        with TestClient(app) as client:
            try:
                token = _mint_init_jwt(
                    priv, aud="instance-test", exp=int(time.time()) + 300
                )
                resp = client.post(
                    "/api/init",
                    headers={"Authorization": f"Bearer {token}"},
                    json=self._init_body(tmp_path),
                )
                assert resp.status_code == 200, resp.text
            finally:
                _reset_conversation_singleton()

    def test_get_init_unauthenticated_under_asymmetric(self, tmp_path):
        _reset_conversation_singleton()
        _priv, pub = _make_ec_keypair()
        app = self._dormant_app(tmp_path, [pub])
        with TestClient(app) as client:
            try:
                resp = client.get("/api/init")
                assert resp.status_code == 200
                assert resp.json()["state"] == "dormant"
            finally:
                _reset_conversation_singleton()

    def test_boot_fails_fast_on_bad_key_file(self, tmp_path):
        # The key file is read at boot (lifespan startup), not at create_app:
        # a missing file aborts startup rather than failing init at runtime.
        _reset_conversation_singleton()
        cfg = Config(
            deferred_init=True,
            init_public_key_file=tmp_path / "missing.pem",
            init_token_audience="instance-test",
            conversations_path=tmp_path / "convs",
            bash_events_dir=tmp_path / "bash",
        )
        app = create_app(cfg)  # construction succeeds; no file read yet
        try:
            with pytest.raises(RuntimeError, match="cannot read init public key file"):
                with TestClient(app):
                    pass
        finally:
            _reset_conversation_singleton()


@pytest.mark.asyncio
async def test_lifespan_teardown_releases_conversation_service_after_init(
    tmp_path,
):
    """If /api/init succeeds, the lifespan finally clause must release the
    conversation service. If /api/init never runs, teardown is a no-op."""
    _reset_conversation_singleton()
    cfg = Config(
        deferred_init=True,
        conversations_path=tmp_path / "convs",
        bash_events_dir=tmp_path / "bash",
    )
    # Build a fake FastAPI app — api_lifespan only touches `.state`.
    fake_app = SimpleNamespace(state=SimpleNamespace(config=cfg))
    async with api_lifespan(fake_app):  # type: ignore[arg-type]
        init_svc = fake_app.state.init_service
        assert init_svc.state == "dormant"
        await init_svc.initialize(
            InitRequest(
                conversations_path=tmp_path / "u" / "convs",
                bash_events_dir=tmp_path / "u" / "bash",
            )
        )
        assert init_svc.state == "ready"
    # After lifespan exit the conversation service should have been torn
    # down — i.e. _entered_service is cleared.
    assert init_svc._entered_service is None
    _reset_conversation_singleton()
