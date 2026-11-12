import unittest

from fido2.server import Fido2Server
from fido2.utils import websafe_decode, websafe_encode
from fido2.webauthn import (
    AttestationConveyancePreference,
    AttestationObject,
    AttestedCredentialData,
    AuthenticationResponse,
    AuthenticatorAssertionResponse,
    AuthenticatorAttestationResponse,
    AuthenticatorData,
    CollectedClientData,
    PublicKeyCredentialParameters,
    PublicKeyCredentialRpEntity,
    PublicKeyCredentialType,
    RegistrationResponse,
    UserVerificationRequirement,
)

from .test_ctap2 import _ATT_CRED_DATA, _CRED_ID


def _params(alg):
    return PublicKeyCredentialParameters(
        type=PublicKeyCredentialType.PUBLIC_KEY, alg=alg
    )


# COSE credential public keys used by the tests. The key material itself is
# not validated by the server, only the structure of the COSE key.
_ES256_KEY = {1: 2, 3: -7, -1: 1, -2: b"\x01" * 32, -3: b"\x02" * 32}
_EDDSA_KEY = {1: 1, 3: -8, -1: 6, -2: b"\x03" * 32}


def _make_registration_response(server, state, public_key, fmt="none", att_stmt=None):
    client_data = CollectedClientData.create(
        CollectedClientData.TYPE.CREATE,
        websafe_decode(state["challenge"]),
        "https://example.com",
    )
    credential_data = AttestedCredentialData.create(b"\0" * 16, _CRED_ID, public_key)
    auth_data = AuthenticatorData.create(
        server.rp.id_hash,
        AuthenticatorData.FLAG.UP | AuthenticatorData.FLAG.AT,
        0,
        credential_data,
    )
    attestation_object = AttestationObject.create(fmt, auth_data, att_stmt or {})
    return RegistrationResponse(
        raw_id=_CRED_ID,
        response=AuthenticatorAttestationResponse(
            client_data=client_data,
            attestation_object=attestation_object,
        ),
    )


class TestPublicKeyCredentialRpEntity(unittest.TestCase):
    def test_id_hash(self):
        rp = PublicKeyCredentialRpEntity(name="Example", id="example.com")
        rp_id_hash = (
            b"\xa3y\xa6\xf6\xee\xaf\xb9\xa5^7\x8c\x11\x804\xe2u\x1eh/"
            b"\xab\x9f-0\xab\x13\xd2\x12U\x86\xce\x19G"
        )
        self.assertEqual(rp.id_hash, rp_id_hash)


USER = {"id": b"user_id", "name": "A. User"}


class TestFido2Server(unittest.TestCase):
    def test_register_begin_rp(self):
        rp = PublicKeyCredentialRpEntity(name="Example", id="example.com")
        server = Fido2Server(rp)

        request, state = server.register_begin(USER)

        self.assertEqual(
            request["publicKey"]["rp"], {"id": "example.com", "name": "Example"}
        )

    def test_register_begin_custom_challenge(self):
        rp = PublicKeyCredentialRpEntity(name="Example", id="example.com")
        server = Fido2Server(rp)

        challenge = b"1234567890123456"
        request, state = server.register_begin(USER, challenge=challenge)

        self.assertEqual(request["publicKey"]["challenge"], websafe_encode(challenge))

    def test_register_begin_custom_challenge_too_short(self):
        rp = PublicKeyCredentialRpEntity(name="Example", id="example.com")
        server = Fido2Server(rp)

        challenge = b"123456789012345"
        with self.assertRaises(ValueError):
            request, state = server.register_begin(USER, challenge=challenge)

    def test_authenticate_complete_invalid_signature(self):
        rp = PublicKeyCredentialRpEntity(name="Example", id="example.com")
        server = Fido2Server(rp)

        state = {
            "challenge": "GAZPACHO!",
            "user_verification": UserVerificationRequirement.PREFERRED,
        }
        client_data = CollectedClientData.create(
            CollectedClientData.TYPE.GET,
            "GAZPACHO!",
            "https://example.com",
        )
        _AUTH_DATA = bytes.fromhex(
            "A379A6F6EEAFB9A55E378C118034E2751E682FAB9F2D30AB13D2125586CE1947010000001D"
        )
        response = AuthenticationResponse(
            raw_id=_CRED_ID,
            response=AuthenticatorAssertionResponse(
                client_data=client_data,
                authenticator_data=AuthenticatorData(_AUTH_DATA),
                signature=b"INVALID",
            ),
        )

        with self.assertRaisesRegex(ValueError, "Invalid signature."):
            server.authenticate_complete(
                state,
                [AttestedCredentialData(_ATT_CRED_DATA)],
                response,
            )


class TestFido2ServerAlgorithmNegotiation(unittest.TestCase):
    def _server(self, algs=(-7,), **kwargs):
        server = Fido2Server(
            PublicKeyCredentialRpEntity(name="Example", id="example.com"), **kwargs
        )
        server.allowed_algorithms = [_params(alg) for alg in algs]
        return server

    def test_register_begin_pins_algorithms_to_state(self):
        server = self._server()
        algs = [_params(-7)]
        server.allowed_algorithms = algs

        request, state = server.register_begin(USER)

        self.assertEqual(state["allowed_algorithms"], [-7])

        # Mutating the assigned list must not affect the issued options/state.
        algs.append(_params(-8))
        algs[0] = _params(-257)
        self.assertEqual(state["allowed_algorithms"], [-7])
        self.assertEqual(
            [p["alg"] for p in request["publicKey"]["pubKeyCredParams"]], [-7]
        )

        # Replacing the server configuration must not affect the state either.
        server.allowed_algorithms = [_params(-8)]
        response = _make_registration_response(server, state, _ES256_KEY)
        server.register_complete(state, response)

    def test_register_complete_allowed_credential_algorithm(self):
        server = self._server()
        request, state = server.register_begin(USER)

        # The attestation signature algorithm differs from the credential
        # public key algorithm; only the latter is checked.
        response = _make_registration_response(
            server,
            state,
            _ES256_KEY,
            fmt="packed",
            att_stmt={"alg": -257, "sig": b"SELF"},
        )
        auth_data = server.register_complete(state, response)
        self.assertEqual(dict(auth_data.credential_data.public_key), _ES256_KEY)

    def test_register_complete_disallowed_credential_algorithm(self):
        server = self._server()
        request, state = server.register_begin(USER)

        # The credential uses EdDSA, which was not offered. The attestation
        # statement claims ES256, which must not be confused with the
        # credential public key algorithm.
        response = _make_registration_response(
            server,
            state,
            _EDDSA_KEY,
            fmt="packed",
            att_stmt={"alg": -7, "sig": b"SELF"},
        )
        with self.assertRaisesRegex(ValueError, "algorithm not allowed"):
            server.register_complete(state, response)

    def test_register_complete_checked_for_all_attestation_types(self):
        for fmt, att_stmt, kwargs in (
            ("none", {}, {}),
            ("packed", {"alg": -7, "sig": b"SELF"}, {}),
            (
                "packed",
                {"alg": -7, "sig": b"CERT", "x5c": [b"CERT"]},
                dict(
                    attestation=AttestationConveyancePreference.DIRECT,
                    verify_attestation=lambda attestation_object, client_data_hash: None,
                ),
            ),
        ):
            with self.subTest(fmt=fmt, att_stmt=att_stmt):
                server = self._server(**kwargs)
                request, state = server.register_begin(USER)
                response = _make_registration_response(
                    server, state, _EDDSA_KEY, fmt=fmt, att_stmt=att_stmt
                )
                with self.assertRaisesRegex(ValueError, "algorithm not allowed"):
                    server.register_complete(state, response)

    def test_register_complete_trusted_attestation_passes(self):
        verified = []

        def verify_attestation(attestation_object, client_data_hash):
            verified.append(attestation_object.fmt)

        server = self._server(
            attestation=AttestationConveyancePreference.DIRECT,
            verify_attestation=verify_attestation,
        )
        request, state = server.register_begin(USER)
        response = _make_registration_response(
            server,
            state,
            _ES256_KEY,
            fmt="packed",
            att_stmt={"alg": -257, "sig": b"CERT", "x5c": [b"CERT"]},
        )
        server.register_complete(state, response)
        self.assertEqual(verified, ["packed"])

    def test_register_complete_curve_mismatch(self):
        server = self._server()
        request, state = server.register_begin(USER)

        key = {**_ES256_KEY, -1: 2}  # ES256 with P-384 curve
        response = _make_registration_response(server, state, key)
        with self.assertRaisesRegex(ValueError, "curve does not match"):
            server.register_complete(state, response)

    def test_register_complete_key_type_mismatch(self):
        server = self._server()
        request, state = server.register_begin(USER)

        key = {**_ES256_KEY, 1: 3}  # ES256 with RSA key type
        response = _make_registration_response(server, state, key)
        with self.assertRaisesRegex(ValueError, "key type does not match"):
            server.register_complete(state, response)

    def test_register_complete_missing_coordinates(self):
        server = self._server()
        request, state = server.register_begin(USER)

        key = {k: v for k, v in _ES256_KEY.items() if k != -3}  # No y-coordinate
        response = _make_registration_response(server, state, key)
        with self.assertRaisesRegex(ValueError, "missing required parameters"):
            server.register_complete(state, response)

    def test_register_complete_private_key_parameter(self):
        server = self._server()
        request, state = server.register_begin(USER)

        key = {**_ES256_KEY, -4: b"\x04" * 32}  # Private key "d" mixed in
        response = _make_registration_response(server, state, key)
        with self.assertRaisesRegex(ValueError, "private key parameters"):
            server.register_complete(state, response)

    def test_register_complete_legacy_state_without_algorithms(self):
        server = self._server()
        request, state = server.register_begin(USER)
        del state["allowed_algorithms"]  # Simulate state from an older version

        response = _make_registration_response(server, state, _EDDSA_KEY)
        with self.assertLogs("fido2.server", level="WARNING"):
            # The current server configuration is not applied retroactively.
            server.register_complete(state, response)
