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
Server-side enforcement of a minimum credProtect level during registration.

Run with: python examples/credprotect_policy.py

The business requirement is that every new credential must be created with
credProtect level 3 (userVerificationRequired). The decision must be based on
the attested extension inside the signed authenticator data, not on the
browser's clientExtensionResults.

Three responses are constructed against the same CredProtectPolicy(3):

1. A genuine authenticator attests credProtect=3 in authenticatorData.
2. Only the client claims credProtect=3; the signed data has no extension.
3. The extension bytes inside the signed authenticator data are tampered with.

Each outcome identifies the exact evidence gap that causes the rejection.
"""

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from fido2 import cbor
from fido2.attestation import Attestation, InvalidSignature
from fido2.cose import ES256
from fido2.extension_policy import (
    CredProtectPolicy,
    ExtensionPolicyError,
)
from fido2.server import Fido2Server
from fido2.webauthn import (
    AttestationConveyancePreference,
    AttestationObject,
    AttestedCredentialData,
    AuthenticatorAttestationResponse,
    AuthenticatorData,
    CollectedClientData,
    PublicKeyCredentialRpEntity,
    RegistrationResponse,
)

RP = PublicKeyCredentialRpEntity(name="Example", id="example.com")
ORIGIN = "https://example.com"
CRED_ID = b"credential_id"
USER = {"id": b"user_id", "name": "A. User"}


def verify_packed(attestation_object, client_data_hash):
    # A relying party would use a configured attestation verifier. Here the
    # built-in packed verifier validates the self-attestation signature.
    Attestation.for_type("packed")().verify(
        attestation_object.att_stmt,
        attestation_object.auth_data,
        client_data_hash,
    )


# Attestation verification is enabled so the authenticator data signature is
# actually checked; the extension policy runs only after it succeeds.
server = Fido2Server(
    RP,
    attestation=AttestationConveyancePreference.DIRECT,
    verify_attestation=verify_packed,
    extension_policy=CredProtectPolicy(3),
)

requested_extensions = {
    "credentialProtectionPolicy": "userVerificationRequired",
    "enforceCredentialProtectionPolicy": True,
}


class SoftwareAuthenticator:
    """Simulates an authenticator issuing packed self-attestations."""

    def __init__(self):
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.credential_key = ES256.from_cryptography_key(self.private_key.public_key())

    def make_response(
        self,
        challenge,
        authenticator_extensions=None,
        client_extension_results=None,
        tamper_signed_extension=False,
    ):
        client_data = CollectedClientData.create(
            CollectedClientData.TYPE.CREATE,
            challenge,
            ORIGIN,
        )
        flags = AuthenticatorData.FLAG.UP | AuthenticatorData.FLAG.AT
        if authenticator_extensions is not None:
            flags |= AuthenticatorData.FLAG.ED

        credential_data = AttestedCredentialData.create(
            b"\0" * 16, CRED_ID, self.credential_key
        )
        auth_data = AuthenticatorData.create(
            RP.id_hash,
            flags,
            0,
            credential_data,
            authenticator_extensions,
        )
        signature = self.private_key.sign(
            bytes(auth_data) + client_data.hash, ec.ECDSA(hashes.SHA256())
        )
        attestation_object = AttestationObject.create(
            "packed",
            auth_data,
            {"alg": ES256.ALGORITHM, "sig": signature},
        )

        if tamper_signed_extension:
            # Rewrite the attested credProtect level from 3 to 2 without
            # re-signing. The CBOR framing stays valid, only the signed byte
            # string changes.
            encoded = cbor.decode(bytes(attestation_object))
            auth_bytes = bytearray(encoded["authData"])
            if auth_bytes[-1] != 3:
                raise ValueError("Expected attested credProtect level 3")
            auth_bytes[-1] = 2
            encoded["authData"] = bytes(auth_bytes)
            attestation_object = AttestationObject(cbor.encode(encoded))

        return RegistrationResponse(
            raw_id=CRED_ID,
            response=AuthenticatorAttestationResponse(
                client_data=client_data,
                attestation_object=attestation_object,
            ),
            client_extension_results=client_extension_results or {},
        )


def run_case(title, state, response):
    print(f"--- {title}")
    try:
        auth_data = server.register_complete(state, response)
    except InvalidSignature:
        print("REJECTED: attestation signature invalid.")
        print(
            "  evidence gap: the extension bytes inside the signed "
            "authenticator data were modified, so the attested evidence is "
            "untrustworthy; the policy is never evaluated."
        )
    except ExtensionPolicyError as e:
        print(f"REJECTED by extension policy: {type(e).__name__}")
        print(f"  evidence gap: {e}")
    else:
        print(
            "ACCEPTED: attested credProtect level = "
            f"{auth_data.extensions['credProtect']}."
        )
    print()


def main():
    authenticator = SoftwareAuthenticator()
    options, state = server.register_begin(USER, extensions=requested_extensions)
    challenge = options["publicKey"]["challenge"]

    # 1. Genuine attested extension: credProtect=3 is part of the signed
    #    authenticator data.
    genuine = authenticator.make_response(
        challenge,
        authenticator_extensions={"credProtect": 3},
    )

    # 2. Client-only claim: the browser reports the value, but the signed
    #    authenticator data contains no extension at all.
    client_only = authenticator.make_response(
        challenge,
        authenticator_extensions=None,
        client_extension_results={"credProtect": 3},
    )

    # 3. Tampered signed bytes: the attested level is silently changed from
    #    3 to 2 after signing.
    tampered = authenticator.make_response(
        challenge,
        authenticator_extensions={"credProtect": 3},
        tamper_signed_extension=True,
    )

    run_case(
        "1. Genuine authenticator extension (credProtect=3 attested)",
        state,
        genuine,
    )
    run_case("2. Only the client claims credProtect=3", state, client_only)
    run_case("3. Signed extension bytes tampered (3 -> 2)", state, tampered)


if __name__ == "__main__":
    main()
