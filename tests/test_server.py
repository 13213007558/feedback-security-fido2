import json
import unittest

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from fido2.cose import ES256
from fido2.server import Fido2Server
from fido2.utils import websafe_encode
from fido2.webauthn import (
    Aaguid,
    AttestationObject,
    AttestedCredentialData,
    AuthenticationResponse,
    AuthenticatorAssertionResponse,
    AuthenticatorAttestationResponse,
    AuthenticatorData,
    CollectedClientData,
    PublicKeyCredentialRpEntity,
    RegistrationResponse,
    UserVerificationRequirement,
)

from .test_ctap2 import _ATT_CRED_DATA, _CRED_ID


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


RP = PublicKeyCredentialRpEntity(name="Example", id="example.com")
RP_ORIGIN = "https://login.example.com"
PARTNER_ORIGIN = "https://embed.partner.example"
TRUSTED_TOP = "https://portal.partner.example"
OTHER_TOP = "https://other-partner.example"
UNTRUSTED_ORIGIN = "https://evil.example.org"


class CrossOriginTestBase(unittest.TestCase):
    def setUp(self):
        # The RP explicitly trusts its own origin and the origin of the embedded
        # partner page, but not an unrelated attacker origin.
        allowed_origins = {RP_ORIGIN, PARTNER_ORIGIN}
        self.server = Fido2Server(
            RP, verify_origin=lambda origin: origin in allowed_origins
        )
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.credential_id = b"credential-identifier"
        self.public_key = ES256.from_cryptography_key(self.private_key.public_key())
        self.credential = AttestedCredentialData.create(
            Aaguid.NONE, self.credential_id, self.public_key
        )

    def _client_data(
        self,
        type_,
        challenge,
        origin=RP_ORIGIN,
        cross_origin=False,
        top_origin=None,
        raw_json=None,
    ):
        if raw_json is not None:
            return CollectedClientData(raw_json)
        return CollectedClientData.create(
            type_,
            challenge,
            origin,
            cross_origin=cross_origin,
            top_origin=top_origin,
        )

    def _attestation(self):
        auth_data = AuthenticatorData.create(
            RP.id_hash,
            AuthenticatorData.FLAG.UP | AuthenticatorData.FLAG.AT,
            0,
            self.credential,
        )
        return AttestationObject.create("none", auth_data, {})

    def _register_response(self, client_data):
        return RegistrationResponse(
            raw_id=self.credential_id,
            response=AuthenticatorAttestationResponse(
                client_data=client_data,
                attestation_object=self._attestation(),
            ),
        )

    def _assertion_response(self, client_data, auth_data):
        signature = self.private_key.sign(
            auth_data + client_data.hash,
            ec.ECDSA(hashes.SHA256()),
        )
        return AuthenticationResponse(
            raw_id=self.credential_id,
            response=AuthenticatorAssertionResponse(
                client_data=client_data,
                authenticator_data=auth_data,
                signature=signature,
            ),
        )

    def _auth_data(self, flags=AuthenticatorData.FLAG.UP):
        return AuthenticatorData.create(RP.id_hash, flags, 1)

    def _assertion(self, client_data):
        auth_data = self._auth_data()
        return self._assertion_response(client_data, auth_data), auth_data


class TestCrossOriginPolicyState(CrossOriginTestBase):
    def test_default_state_is_same_origin_only(self):
        _, state = self.server.register_begin(USER)
        self.assertIs(state["cross_origin"], False)
        self.assertEqual(set(state["top_origins"]), set())

        _, auth_state = self.server.authenticate_begin()
        self.assertIs(auth_state["cross_origin"], False)
        self.assertEqual(set(auth_state["top_origins"]), set())

    def test_cross_origin_policy_stored_in_state(self):
        _, state = self.server.register_begin(
            USER, cross_origin=True, top_origins=[TRUSTED_TOP]
        )
        self.assertIs(state["cross_origin"], True)
        self.assertEqual(state["top_origins"], frozenset({TRUSTED_TOP}))

    def test_cross_origin_requires_top_origins(self):
        with self.assertRaisesRegex(ValueError, "top_origin"):
            self.server.register_begin(USER, cross_origin=True)
        with self.assertRaisesRegex(ValueError, "top_origin"):
            self.server.authenticate_begin(cross_origin=True)

    def test_cross_origin_must_be_boolean(self):
        # The string "false" must not be coerced to a truthy/falsy value.
        with self.assertRaisesRegex(TypeError, "boolean"):
            self.server.register_begin(USER, cross_origin="false")
        with self.assertRaisesRegex(TypeError, "boolean"):
            self.server.authenticate_begin(cross_origin="true")
        with self.assertRaisesRegex(TypeError, "boolean"):
            self.server.register_begin(USER, cross_origin=1)

    def test_top_origins_must_be_strings(self):
        with self.assertRaisesRegex(TypeError, "origin strings"):
            self.server.register_begin(
                USER, cross_origin=True, top_origins=[TRUSTED_TOP, 42]
            )

    def test_parallel_states_keep_separate_policies(self):
        _, strict_state = self.server.authenticate_begin()
        _, embedded_state = self.server.authenticate_begin(
            cross_origin=True, top_origins=[TRUSTED_TOP]
        )
        self.assertIs(strict_state["cross_origin"], False)
        self.assertIs(embedded_state["cross_origin"], True)
        self.assertEqual(strict_state["top_origins"], frozenset())
        self.assertEqual(embedded_state["top_origins"], frozenset({TRUSTED_TOP}))
        self.assertNotEqual(strict_state["challenge"], embedded_state["challenge"])


class TestSameOriginFlow(CrossOriginTestBase):
    def test_register_same_origin(self):
        options, state = self.server.register_begin(USER)
        client_data = self._client_data(
            CollectedClientData.TYPE.CREATE,
            options.public_key.challenge,
        )
        auth_data = self.server.register_complete(
            state, self._register_response(client_data)
        )
        self.assertTrue(auth_data.is_user_present())

    def test_authenticate_same_origin(self):
        options, state = self.server.authenticate_begin([self.credential])
        client_data = self._client_data(
            CollectedClientData.TYPE.GET, options.public_key.challenge
        )
        response, _ = self._assertion(client_data)
        cred = self.server.authenticate_complete(state, [self.credential], response)
        self.assertEqual(cred.credential_id, self.credential_id)

    def test_invalid_origin_rejected(self):
        options, state = self.server.register_begin(USER)
        client_data = self._client_data(
            CollectedClientData.TYPE.CREATE,
            options.public_key.challenge,
            origin=UNTRUSTED_ORIGIN,
        )
        with self.assertRaisesRegex(ValueError, "Invalid origin"):
            self.server.register_complete(state, self._register_response(client_data))


class TestCrossOriginFlow(CrossOriginTestBase):
    def _begin(self, method, *args, top_origins=(TRUSTED_TOP,), **kwargs):
        begin = getattr(self.server, method)
        return begin(*args, cross_origin=True, top_origins=list(top_origins), **kwargs)

    def test_register_authorized_embedding(self):
        options, state = self._begin("register_begin", USER)
        client_data = self._client_data(
            CollectedClientData.TYPE.CREATE,
            options.public_key.challenge,
            origin=PARTNER_ORIGIN,
            cross_origin=True,
            top_origin=TRUSTED_TOP,
        )
        auth_data = self.server.register_complete(
            state, self._register_response(client_data)
        )
        self.assertTrue(auth_data.is_user_present())

    def test_authenticate_authorized_embedding(self):
        options, state = self._begin("authenticate_begin", [self.credential])
        client_data = self._client_data(
            CollectedClientData.TYPE.GET,
            options.public_key.challenge,
            origin=PARTNER_ORIGIN,
            cross_origin=True,
            top_origin=TRUSTED_TOP,
        )
        response, _ = self._assertion(client_data)
        cred = self.server.authenticate_complete(state, [self.credential], response)
        self.assertEqual(cred.credential_id, self.credential_id)

    def test_cross_origin_rejected_by_default(self):
        options, state = self.server.authenticate_begin([self.credential])
        client_data = self._client_data(
            CollectedClientData.TYPE.GET,
            options.public_key.challenge,
            origin=PARTNER_ORIGIN,
            cross_origin=True,
            top_origin=TRUSTED_TOP,
        )
        response, _ = self._assertion(client_data)
        with self.assertRaisesRegex(ValueError, "not permitted"):
            self.server.authenticate_complete(state, [self.credential], response)

    def test_forged_top_origin_rejected(self):
        options, state = self._begin("authenticate_begin", [self.credential])
        client_data = self._client_data(
            CollectedClientData.TYPE.GET,
            options.public_key.challenge,
            origin=PARTNER_ORIGIN,
            cross_origin=True,
            top_origin=OTHER_TOP,
        )
        response, _ = self._assertion(client_data)
        with self.assertRaisesRegex(ValueError, "Invalid topOrigin"):
            self.server.authenticate_complete(state, [self.credential], response)

    def test_missing_top_origin_rejected(self):
        options, state = self._begin("authenticate_begin", [self.credential])
        client_data = self._client_data(
            CollectedClientData.TYPE.GET,
            options.public_key.challenge,
            origin=PARTNER_ORIGIN,
            cross_origin=True,
        )
        response, _ = self._assertion(client_data)
        with self.assertRaisesRegex(ValueError, "topOrigin"):
            self.server.authenticate_complete(state, [self.credential], response)

    def test_response_value_never_builds_whitelist(self):
        # A topOrigin seen in the response is not trusted just because the
        # browser sent it; only the caller-provided policy set is used.
        options, state = self._begin(
            "authenticate_begin",
            [self.credential],
            top_origins=(TRUSTED_TOP,),
        )
        self.assertNotIn(OTHER_TOP, state["top_origins"])
        client_data = self._client_data(
            CollectedClientData.TYPE.GET,
            options.public_key.challenge,
            origin=PARTNER_ORIGIN,
            cross_origin=True,
            top_origin=OTHER_TOP,
        )
        response, _ = self._assertion(client_data)
        with self.assertRaisesRegex(ValueError, "Invalid topOrigin"):
            self.server.authenticate_complete(state, [self.credential], response)

    def test_origin_and_top_origin_checked_independently(self):
        # A trusted topOrigin does not rescue an origin that fails the RP
        # check; the two checks never borrow trust from each other.
        options, state = self._begin("authenticate_begin", [self.credential])
        client_data = self._client_data(
            CollectedClientData.TYPE.GET,
            options.public_key.challenge,
            origin=UNTRUSTED_ORIGIN,
            cross_origin=True,
            top_origin=TRUSTED_TOP,
        )
        response, _ = self._assertion(client_data)
        with self.assertRaisesRegex(ValueError, "Invalid origin"):
            self.server.authenticate_complete(state, [self.credential], response)

    def test_string_false_cross_origin_not_coerced(self):
        options, state = self._begin("authenticate_begin", [self.credential])
        raw = json.dumps(
            {
                "type": CollectedClientData.TYPE.GET,
                "challenge": websafe_encode(options.public_key.challenge),
                "origin": PARTNER_ORIGIN,
                "crossOrigin": "false",
                "topOrigin": TRUSTED_TOP,
            }
        ).encode()
        client_data = self._client_data(None, None, raw_json=raw)
        response, _ = self._assertion(client_data)
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.server.authenticate_complete(state, [self.credential], response)

    def test_integer_cross_origin_not_coerced(self):
        options, state = self.server.authenticate_begin([self.credential])
        raw = json.dumps(
            {
                "type": CollectedClientData.TYPE.GET,
                "challenge": websafe_encode(options.public_key.challenge),
                "origin": RP_ORIGIN,
                "crossOrigin": 1,
            }
        ).encode()
        client_data = self._client_data(None, None, raw_json=raw)
        response, _ = self._assertion(client_data)
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.server.authenticate_complete(state, [self.credential], response)


class TestLegacyStateCompatibility(CrossOriginTestBase):
    def _legacy_state(self, challenge):
        # State as produced by library versions without cross-origin fields.
        return {
            "challenge": websafe_encode(challenge),
            "user_verification": UserVerificationRequirement.PREFERRED,
        }

    def test_legacy_state_accepts_same_origin(self):
        options, _ = self.server.authenticate_begin([self.credential])
        state = self._legacy_state(options.public_key.challenge)
        client_data = self._client_data(
            CollectedClientData.TYPE.GET, options.public_key.challenge
        )
        response, _ = self._assertion(client_data)
        cred = self.server.authenticate_complete(state, [self.credential], response)
        self.assertEqual(cred.credential_id, self.credential_id)

    def test_legacy_state_does_not_silently_allow_cross_origin(self):
        options, _ = self.server.authenticate_begin([self.credential])
        state = self._legacy_state(options.public_key.challenge)
        client_data = self._client_data(
            CollectedClientData.TYPE.GET,
            options.public_key.challenge,
            origin=PARTNER_ORIGIN,
            cross_origin=True,
            top_origin=TRUSTED_TOP,
        )
        response, _ = self._assertion(client_data)
        with self.assertRaisesRegex(ValueError, "not permitted"):
            self.server.authenticate_complete(state, [self.credential], response)


class TestInterleavedFlows(CrossOriginTestBase):
    def test_interleaved_bounce_between_two_parallel_logins(self):
        # Login A is configured for embedding; login B is same-origin only.
        options_a, state_a = self.server.authenticate_begin(
            [self.credential],
            cross_origin=True,
            top_origins=[TRUSTED_TOP],
        )
        options_b, state_b = self.server.authenticate_begin([self.credential])

        # A cross-origin assertion bounced at the strict login B is rejected.
        cross_data = self._client_data(
            CollectedClientData.TYPE.GET,
            options_b.public_key.challenge,
            origin=PARTNER_ORIGIN,
            cross_origin=True,
            top_origin=TRUSTED_TOP,
        )
        response_b, _ = self._assertion(cross_data)
        with self.assertRaisesRegex(ValueError, "not permitted"):
            self.server.authenticate_complete(state_b, [self.credential], response_b)

        # The same embedded login A still accepts its own cross-origin
        # assertion; the two states never share configuration.
        cross_data_a = self._client_data(
            CollectedClientData.TYPE.GET,
            options_a.public_key.challenge,
            origin=PARTNER_ORIGIN,
            cross_origin=True,
            top_origin=TRUSTED_TOP,
        )
        response_a, _ = self._assertion(cross_data_a)
        cred = self.server.authenticate_complete(state_a, [self.credential], response_a)
        self.assertEqual(cred.credential_id, self.credential_id)

    def test_same_origin_response_under_cross_origin_policy(self):
        # Permitting cross-origin does not require it: a plain same-origin
        # assertion against an embedding-enabled state still succeeds.
        options, state = self.server.authenticate_begin(
            [self.credential],
            cross_origin=True,
            top_origins=[TRUSTED_TOP],
        )
        client_data = self._client_data(
            CollectedClientData.TYPE.GET, options.public_key.challenge
        )
        response, _ = self._assertion(client_data)
        cred = self.server.authenticate_complete(state, [self.credential], response)
        self.assertEqual(cred.credential_id, self.credential_id)


class TestRawClientDataSignatureBinding(CrossOriginTestBase):
    def _non_canonical_client_data(self, challenge, origin=PARTNER_ORIGIN):
        data = {
            "type": CollectedClientData.TYPE.GET,
            "challenge": websafe_encode(challenge),
            "origin": origin,
            "crossOrigin": origin != RP_ORIGIN,
            "topOrigin": TRUSTED_TOP if origin != RP_ORIGIN else None,
        }
        if data["topOrigin"] is None:
            del data["topOrigin"]
        # Pretty-printed JSON with whitespace: not the compact canonical form
        # produced by CollectedClientData.create().
        return CollectedClientData(json.dumps(data, indent=2).encode())

    def test_signature_verified_over_original_bytes(self):
        options, state = self.server.authenticate_begin(
            [self.credential],
            cross_origin=True,
            top_origins=[TRUSTED_TOP],
        )
        client_data = self._non_canonical_client_data(options.public_key.challenge)
        auth_data = self._auth_data()
        # Signature binds the exact bytes received, including whitespace.
        signature = self.private_key.sign(
            auth_data + client_data.hash, ec.ECDSA(hashes.SHA256())
        )
        response = AuthenticationResponse(
            raw_id=self.credential_id,
            response=AuthenticatorAssertionResponse(
                client_data=client_data,
                authenticator_data=auth_data,
                signature=signature,
            ),
        )
        cred = self.server.authenticate_complete(state, [self.credential], response)
        self.assertEqual(cred.credential_id, self.credential_id)

    def test_tampered_raw_bytes_break_signature(self):
        # A byte-level change that preserves the parsed JSON content must still
        # invalidate the signature, proving the server verifies the original
        # clientDataJSON bytes rather than a re-serialized copy.
        options, state = self.server.authenticate_begin([self.credential])
        client_data = self._non_canonical_client_data(
            options.public_key.challenge, origin=RP_ORIGIN
        )
        auth_data = self._auth_data()
        signature = self.private_key.sign(
            auth_data + client_data.hash, ec.ECDSA(hashes.SHA256())
        )
        tampered = CollectedClientData(client_data.replace(b'"type":', b'"type" :', 1))
        self.assertEqual(tampered.type, client_data.type)  # Same parsed content
        response = AuthenticationResponse(
            raw_id=self.credential_id,
            response=AuthenticatorAssertionResponse(
                client_data=tampered,
                authenticator_data=auth_data,
                signature=signature,
            ),
        )
        with self.assertRaisesRegex(ValueError, "Invalid signature"):
            self.server.authenticate_complete(state, [self.credential], response)
