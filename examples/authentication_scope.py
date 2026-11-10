# Copyright (c) 2024 Yubico AB
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
Scope and user-binding guarantees of Fido2Server.authenticate_complete.

This example does not need a physical authenticator: credentials and assertions
are generated in software. It registers one credential for each of two accounts
and then interweaves them to show that:

  1. With allowCredentials, only the IDs pinned at authenticate_begin can
     authenticate. A correctly signed assertion from a credential outside that
     set is rejected even when its public key is among the candidates.
  2. For a discoverable credential flow (no allowCredentials), the server binds
     the assertion to a user from trusted data. A credential that signs correctly
     but returns another account's userHandle is rejected.

The "state" object returned by authenticate_begin is what stores the scope. In a
web application it is typically saved in the user's session and handed back,
unchanged, to authenticate_complete.
"""

import hashlib

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from fido2.cose import ES256
from fido2.server import Fido2Server
from fido2.utils import websafe_encode
from fido2.webauthn import (
    Aaguid,
    AttestedCredentialData,
    AuthenticationResponse,
    AuthenticatorAssertionResponse,
    AuthenticatorData,
    CollectedClientData,
    PublicKeyCredentialRpEntity,
)

rp = PublicKeyCredentialRpEntity(name="Example RP", id="example.com")
server = Fido2Server(rp)
rp_id_hash = hashlib.sha256(rp.id.encode()).digest()
ORIGIN = "https://example.com"


def make_account(user_handle, credential_id):
    """A registered account: its credential, the matching private key, the user."""
    private_key = ec.generate_private_key(ec.SECP256R1())
    credential = AttestedCredentialData.create(
        Aaguid.NONE,
        credential_id,
        ES256.from_cryptography_key(private_key.public_key()),
    )
    return {
        "user_handle": user_handle,
        "credential": credential,
        "private_key": private_key,
    }


def make_assertion(account, challenge, *, user_handle):
    """Produce a correctly signed authentication response for an account."""
    client_data = CollectedClientData.create(
        CollectedClientData.TYPE.GET, websafe_encode(challenge), ORIGIN
    )
    authenticator_data = AuthenticatorData.create(
        rp_id_hash, AuthenticatorData.FLAG.UP, 1
    )
    signature = account["private_key"].sign(
        authenticator_data + client_data.hash, ec.ECDSA(hashes.SHA256())
    )
    return AuthenticationResponse(
        raw_id=account["credential"].credential_id,
        response=AuthenticatorAssertionResponse(
            client_data=client_data,
            authenticator_data=authenticator_data,
            signature=signature,
            user_handle=user_handle,
        ),
    )


account_a = make_account(b"user-a", b"credential-id-a")
account_b = make_account(b"user-b", b"credential-id-b")

# The server-side store: every registered credential for every account, plus the
# trusted credential-ID -> user-handle relationship used for discoverable logins.
all_credentials = [account_a["credential"], account_b["credential"]]
credential_users = {
    account_a["credential"].credential_id: account_a["user_handle"],
    account_b["credential"].credential_id: account_b["user_handle"],
}

# ---------------------------------------------------------------------------
# Flow 1: allowCredentials pins the allowed credential IDs in the state.
# ---------------------------------------------------------------------------
print("== allowCredentials flow ==")

# This ceremony is only allowed to use account A's credential. The returned state
# (which a web app stores in the session) records exactly that scope.
challenge = b"0" * 32
options, state = server.authenticate_begin(
    [account_a["credential"]], challenge=challenge
)
print("Pinned scope stored in state:", state["allow_credentials"])

# The legitimate assertion from A succeeds.
good = make_assertion(account_a, challenge, user_handle=account_a["user_handle"])
result = server.authenticate_complete(state, all_credentials, good)
print("Account A authenticated:", result.credential.credential_id)

# Attack: B's credential produces a fully valid signature, but it was not part of
# the allowCredentials pinned for this ceremony. Passing the whole credential store
# as candidates does not help: the state, not the candidate list, defines scope.
attack = make_assertion(account_b, challenge, user_handle=account_b["user_handle"])
try:
    server.authenticate_complete(state, all_credentials, attack)
except ValueError as e:
    print("Out-of-scope credential rejected:", e)

# Mutating the original list after begin() also cannot widen the pinned scope.
allowed = [account_a["credential"]]
_, state = server.authenticate_begin(allowed, challenge=challenge)
allowed.append(account_b["credential"])
try:
    server.authenticate_complete(state, all_credentials, attack)
except ValueError as e:
    print("Post-begin list mutation rejected:", e)

# ---------------------------------------------------------------------------
# Flow 2: discoverable credentials (no allowCredentials) are bound to a user.
# ---------------------------------------------------------------------------
print("\n== discoverable credential flow ==")

# No credentials are sent to the authenticator, so there is nothing to pin. The
# server instead supplies trusted user information at completion time.
challenge = b"1" * 32
_, state = server.authenticate_begin(challenge=challenge)
print("Pinned scope stored in state:", state["allow_credentials"])

# A well-formed discoverable assertion from A, carrying A's own userHandle, works.
good = make_assertion(account_a, challenge, user_handle=account_a["user_handle"])
result = server.authenticate_complete(
    state,
    all_credentials,
    good,
    credential_users=credential_users,
)
print("Account A authenticated, resolved user:", result.user_handle)

# Attack: A's credential signs correctly but the response claims account B's
# userHandle. The trusted mapping says this credential belongs to A, so the
# mismatched handle is rejected instead of authenticating the wrong account.
_, state = server.authenticate_begin(challenge=challenge)
spoofed = make_assertion(account_a, challenge, user_handle=account_b["user_handle"])
try:
    server.authenticate_complete(
        state,
        all_credentials,
        spoofed,
        credential_users=credential_users,
    )
except ValueError as e:
    print("Cross-account userHandle rejected:", e)

# The same protection applies when the server simply names the expected user.
_, state = server.authenticate_begin(challenge=challenge)
try:
    server.authenticate_complete(
        state,
        all_credentials,
        spoofed,
        expected_user_handle=account_a["user_handle"],
    )
except ValueError as e:
    print("userHandle != expected user rejected:", e)
