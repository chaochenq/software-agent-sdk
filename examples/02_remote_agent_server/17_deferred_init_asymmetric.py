"""Deferred-init (warm-pool) mode with ASYMMETRIC init authentication.

This is the asymmetric counterpart to ``16_deferred_init.py``. Instead of the
orchestrator and the server sharing a symmetric bootstrap secret, the dormant
server is configured with a *public key* and trusts a short-lived JWT signed by
the matching *private key*. The orchestrator holds the private key; the server
only ever holds the (non-secret) public key, so a compromised server instance
cannot forge init calls. The frontend never sees the private key — it only
receives the session API keys minted by init.

How it maps to a real deployment:
  * The instance boots with ``OH_INIT_PUBLIC_KEY_FILE`` pointing at a PEM file of
    the trusted public key(s) — typically a mounted secret.
  * The expected token audience binds tokens to this specific instance. By
    default the server reads it from the ``AGENT_SERVER_NAME`` env var, which the
    orchestrator sets per instance (override the source via
    ``OH_INIT_TOKEN_AUDIENCE_ENV``, or set a literal via
    ``OH_INIT_TOKEN_AUDIENCE``). This demo sets ``AGENT_SERVER_NAME``.
  * The orchestrator mints a JWT with ``aud`` = that instance and a short
    ``exp``, and POSTs it as ``Authorization: Bearer <jwt>`` to ``/api/init``.

Lifecycle demonstrated here:
  1. Server starts in dormant mode (configured with the public key).
  2. ``POST /api/init`` WITHOUT a token returns 401 (auth is enforced).
  3. ``POST /api/init`` with a valid signed JWT → server transitions to ready.
  4. A conversation runs normally on the now-ready server.
"""

import os
import tempfile
import time
from uuid import UUID

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from joserfc import jwk, jwt
from scripts.utils import ManagedAPIServer

from openhands.sdk import get_logger


logger = get_logger(__name__)

# ── LLM config ──────────────────────────────────────────────────────────────

api_key = os.getenv("LLM_API_KEY")
assert api_key is not None, "LLM_API_KEY environment variable is not set."
llm_model = os.getenv("LLM_MODEL", "gpt-5.5")
llm_base_url = os.getenv("LLM_BASE_URL")

# ── Asymmetric init key material ─────────────────────────────────────────────
# The orchestrator owns the PRIVATE key; the agent server only trusts the PUBLIC
# key. In production these are managed/rotated by the orchestrator; here we
# generate an ephemeral EC P-256 (ES256) keypair to make the example
# self-contained.

_signing_key = ec.generate_private_key(ec.SECP256R1())
PRIVATE_PEM = _signing_key.private_bytes(
    serialization.Encoding.PEM,
    serialization.PrivateFormat.PKCS8,
    serialization.NoEncryption(),
).decode()
PUBLIC_PEM = (
    _signing_key.public_key()
    .public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    .decode()
)
# The agent server trusts public key(s) loaded from a PEM file. In production
# this is typically a mounted secret; here we write the ephemeral public key to
# a temp file and point OH_INIT_PUBLIC_KEY_FILE at it. The file may hold several
# concatenated PEM blocks to trust multiple keys (rotation).
_key_dir = tempfile.mkdtemp(prefix="deferred_init_asym_keys_")
PUBLIC_KEY_FILE = os.path.join(_key_dir, "init_public_key.pem")
with open(PUBLIC_KEY_FILE, "w") as _f:
    _f.write(PUBLIC_PEM)

# The audience binds a token to this specific instance (prevents cross-instance
# replay). By default the server reads it from the AGENT_SERVER_NAME env var,
# which the orchestrator sets per instance; an identical deploy spec then gives
# each instance a distinct audience. (Override the source var name via
# OH_INIT_TOKEN_AUDIENCE_ENV.)
INSTANCE_AUDIENCE = "demo-instance-001"


def mint_init_token() -> str:
    """Mint a short-lived init JWT, signed with the orchestrator's private key."""
    now = int(time.time())
    return jwt.encode(
        {"alg": "ES256"},
        {
            "iss": "orchestrator-demo",
            "aud": INSTANCE_AUDIENCE,
            "iat": now,
            "exp": now + 120,  # short-lived: init happens right after matching
        },
        jwk.import_key(PRIVATE_PEM.encode(), "EC"),
    )


# ── Server lifecycle ─────────────────────────────────────────────────────────

with ManagedAPIServer(
    port=8004,
    extra_env={
        "OH_DEFERRED_INIT": "true",
        # OH_SECRET_KEY still configures the encryption cipher, but it is NOT an
        # init credential once a public key is configured (asymmetric wins).
        "OH_SECRET_KEY": "demo-cipher-key-only-32-bytes-ok!",
        "OH_INIT_PUBLIC_KEY_FILE": PUBLIC_KEY_FILE,
        # The server reads its expected audience from AGENT_SERVER_NAME by
        # default; the orchestrator sets it per instance. (No need to set
        # OH_INIT_TOKEN_AUDIENCE_ENV unless you want a different source var.)
        "AGENT_SERVER_NAME": INSTANCE_AUDIENCE,
        "TMUX_TMPDIR": "/tmp/oh-tmux-deferred-asym",
    },
) as server:
    client = httpx.Client(base_url=server.base_url, timeout=120.0)

    try:
        # ── 1. Confirm dormant state ─────────────────────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("📊 Step 1: checking initial (dormant) state")
        logger.info("=" * 60)

        resp = client.get("/api/init")
        assert resp.status_code == 200, f"GET /api/init failed: {resp.text}"
        assert resp.json()["state"] == "dormant", resp.json()
        logger.info("✅ Server is dormant (public key configured)")

        # ── 2. Init without a token is rejected ──────────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("🔒 Step 2: POST /api/init without a token returns 401")
        logger.info("=" * 60)

        resp = client.post("/api/init", json={})
        assert resp.status_code == 401, (
            f"Expected 401 without a token, got {resp.status_code}"
        )
        logger.info("✅ Unauthenticated init correctly rejected (401)")

        # ── 3. Activate via POST /api/init with a signed JWT ─────────────────
        logger.info("\n" + "=" * 60)
        logger.info("🚀 Step 3: activating via POST /api/init (Bearer JWT)")
        logger.info("=" * 60)

        temp_workspace_dir = tempfile.mkdtemp(prefix="deferred_init_asym_demo_")

        # In a real warm-pool deployment, credentials that the server shouldn't
        # have at cold-start (e.g., the user's LLM API key) arrive here.
        llm_env: dict[str, str] = {"LLM_API_KEY": api_key}
        if llm_base_url:
            llm_env["LLM_BASE_URL"] = llm_base_url

        init_body: dict = {"env": llm_env}

        resp = client.post(
            "/api/init",
            json=init_body,
            headers={"Authorization": f"Bearer {mint_init_token()}"},
        )
        assert resp.status_code == 200, f"POST /api/init failed: {resp.text}"
        assert resp.json()["state"] == "ready", resp.json()
        logger.info("✅ Server is now ready (asymmetric init succeeded)")

        # ── 4. Run a conversation on the now-ready server ────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("🤖 Step 4: running a conversation on the ready server")
        logger.info("=" * 60)

        llm_config: dict[str, str] = {"model": llm_model, "api_key": api_key}
        if llm_base_url:
            llm_config["base_url"] = llm_base_url

        start_request: dict = {
            "agent": {
                "kind": "Agent",
                "llm": llm_config,
                "tools": [],
            },
            "workspace": {"working_dir": temp_workspace_dir},
            "initial_message": {
                "role": "user",
                "content": [{"type": "text", "text": "Reply with just the number 42."}],
                "run": True,
            },
        }

        resp = client.post("/api/conversations", json=start_request)
        assert resp.status_code == 201, f"Start conversation failed: {resp.text}"
        conversation_id = UUID(resp.json()["id"])
        logger.info(f"✅ Conversation started: {conversation_id}")

        # Poll until the agent finishes.
        max_wait = 120
        elapsed = 0
        execution_status = "unknown"
        while elapsed < max_wait:
            resp = client.get(f"/api/conversations/{conversation_id}")
            assert resp.status_code == 200
            data = resp.json()
            execution_status = data.get("execution_status", "unknown")
            # Terminal states per ConversationExecutionStatus.is_terminal().
            if execution_status in ("finished", "error", "stuck"):
                break
            logger.info(f"   status: {execution_status} ({elapsed}s elapsed)")
            time.sleep(2)
            elapsed += 2

        logger.info(f"✅ Conversation finished — status: {execution_status}")
        assert execution_status == "finished", (
            f"Unexpected final status: {execution_status}"
        )

        resp = client.get(f"/api/conversations/{conversation_id}/agent_final_response")
        if resp.status_code == 200:
            agent_response = resp.json().get("response", "")
            logger.info(f"   Agent response: {agent_response!r}")

        # Collect cost metrics.
        accumulated_cost = 0.0
        resp = client.get(f"/api/conversations/{conversation_id}")
        if resp.status_code == 200:
            stats = resp.json().get("stats") or {}
            usage_to_metrics = stats.get("usage_to_metrics") or {}
            accumulated_cost = sum(
                m.get("accumulated_cost", 0.0) for m in usage_to_metrics.values()
            )

        client.delete(f"/api/conversations/{conversation_id}")
        logger.info("   Conversation deleted")

        logger.info("\n" + "=" * 60)
        logger.info("🎉 Asymmetric deferred-init example completed successfully!")
        logger.info("=" * 60)

        print(f"EXAMPLE_COST: {accumulated_cost}")

    finally:
        client.close()
