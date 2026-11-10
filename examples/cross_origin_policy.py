# Copyright (c) 2026 Yubico AB
# All rights reserved.
#
#   Redistribution and use in source and binary forms, with or
#   without modification, are permitted provided that the following
#   conditions are met:
#
#    1. Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#    2. Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Example: explicit cross-origin (embedded page) policy for WebAuthn.

A login page is sometimes embedded inside a partner website, e.g. inside an
<iframe>. In that case the browser sets ``crossOrigin: true`` and reports the
embedding site as the ``topOrigin`` in CollectedClientData. This example shows
how the relying party opts in to such flows, and makes visible which origins
are still rejected afterwards.

Run it directly, no hardware or network access is required:

    python examples/cross_origin_policy.py

The example builds locally signed WebAuthn assertions (an ES256 key generated
in memory) and feeds them to Fido2Server.authenticate_complete, mirroring what
a browser would send.
"""

from __future__ import annotations

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from fido2.cose import ES256
from fido2.server import Fido2Server
from fido2.webauthn import (
    Aaguid,
    AttestedCredentialData,
    AuthenticationResponse,
    AuthenticatorAssertionResponse,
    AuthenticatorData,
    CollectedClientData,
    PublicKeyCredentialRpEntity,
)

RP_ID = "example.com"
# Origins from which the WebAuthn call itself may be made. The embedded login
# page is served from a dedicated host, which the RP explicitly trusts here.
RP_ORIGIN = "https://login.example.com"
EMBED_ORIGIN = "https://widget.partner.example"
ALLOWED_ORIGINS = {RP_ORIGIN, EMBED_ORIGIN}

# Top-level (embedding) sites the RP is willing to be framed by.
TRUSTED_PARTNER = "https://portal.partner.example"

# Origins used below to show that embedding one partner does not trust others.
UNTRUSTED_PARTNER = "https://other-partner.example"
ATTACKER_ORIGIN = "https://evil.example.org"


def make_server() -> (
    tuple[Fido2Server, AttestedCredentialData, ec.EllipticCurvePrivateKey]
):
    rp = PublicKeyCredentialRpEntity(name="Example RP", id=RP_ID)
    # verify_origin decides which effective origins (the frame that actually
    # calls navigator.credentials.get) are acceptable. The cross-origin policy
    # below is a separate, additional check on the top-level origin.
    server = Fido2Server(rp, verify_origin=lambda origin: origin in ALLOWED_ORIGINS)

    private_key = ec.generate_private_key(ec.SECP256R1())
    credential = AttestedCredentialData.create(
        Aaguid.NONE,
        b"example-credential-id",
        ES256.from_cryptography_key(private_key.public_key()),
    )
    return server, credential, private_key


def assert_credential(server, credential, private_key, state, client_data):
    auth_data = AuthenticatorData.create(
        rp_id_hash=server.rp.id_hash, flags=AuthenticatorData.FLAG.UP, counter=1
    )
    signature = private_key.sign(
        auth_data + client_data.hash, ec.ECDSA(hashes.SHA256())
    )
    response = AuthenticationResponse(
        raw_id=credential.credential_id,
        response=AuthenticatorAssertionResponse(
            client_data=client_data,
            authenticator_data=auth_data,
            signature=signature,
        ),
    )
    return server.authenticate_complete(state, [credential], response)


def attempt(label, server, credential, private_key, state, **client_kwargs):
    challenge = state["challenge"]
    client_data = CollectedClientData.create(
        CollectedClientData.TYPE.GET, challenge, **client_kwargs
    )
    try:
        assert_credential(server, credential, private_key, state, client_data)
    except ValueError as error:
        print(f"  REJECTED  {label}: {error}")
    else:
        print(f"  ACCEPTED  {label}")


def main():
    server, credential, private_key = make_server()

    print("=== Same-origin-only policy (the default) ===")
    _, strict_state = server.authenticate_begin([credential])

    attempt(
        "plain page served by the RP itself",
        server,
        credential,
        private_key,
        strict_state,
        origin=RP_ORIGIN,
    )
    attempt(
        "partner frame, crossOrigin=true (not opted in)",
        server,
        credential,
        private_key,
        strict_state,
        origin=EMBED_ORIGIN,
        cross_origin=True,
        top_origin=TRUSTED_PARTNER,
    )

    print()
    print("=== Embedding enabled for one trusted partner ===")
    print(f"  trusted topOrigin allowlist: {{{TRUSTED_PARTNER!r}}}")
    _, embedded_state = server.authenticate_begin(
        [credential],
        cross_origin=True,
        top_origins=[TRUSTED_PARTNER],
    )

    attempt(
        "the approved partner embeds the login page",
        server,
        credential,
        private_key,
        embedded_state,
        origin=EMBED_ORIGIN,
        cross_origin=True,
        top_origin=TRUSTED_PARTNER,
    )
    attempt(
        "plain RP page still works (crossOrigin stays optional)",
        server,
        credential,
        private_key,
        embedded_state,
        origin=RP_ORIGIN,
    )
    attempt(
        "forged topOrigin from a different site",
        server,
        credential,
        private_key,
        embedded_state,
        origin=EMBED_ORIGIN,
        cross_origin=True,
        top_origin=UNTRUSTED_PARTNER,
    )
    attempt(
        "crossOrigin=true with no topOrigin at all",
        server,
        credential,
        private_key,
        embedded_state,
        origin=EMBED_ORIGIN,
        cross_origin=True,
    )
    attempt(
        "trusted topOrigin cannot rescue an untrusted frame origin",
        server,
        credential,
        private_key,
        embedded_state,
        origin=ATTACKER_ORIGIN,
        cross_origin=True,
        top_origin=TRUSTED_PARTNER,
    )

    print()
    print("Observations:")
    print("  * Allowing one partner topOrigin never widens the frame-origin check.")
    print("  * A topOrigin seen in a response is not trusted unless it was supplied")
    print("    in top_origins when the operation began.")
    print("  * Same-origin-only states reject crossOrigin=true outright, including")
    print("    legacy states that predate these policy fields.")


if __name__ == "__main__":
    main()
