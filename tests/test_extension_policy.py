import unittest
from types import MappingProxyType

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from fido2 import cbor
from fido2.attestation import Attestation, InvalidSignature
from fido2.extension_policy import (
    AbstractExtensionPolicy,
    CredProtectPolicy,
    InsufficientExtensionLevelError,
    InvalidExtensionValueError,
    MissingExtensionError,
    RegistrationExtensionEvidence,
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


def _verify_packed(attestation_object, client_data_hash):
    Attestation.for_type("packed")().verify(
        attestation_object.att_stmt,
        attestation_object.auth_data,
        client_data_hash,
    )


def _make_server(policy=None):
    return Fido2Server(
        RP,
        attestation=AttestationConveyancePreference.DIRECT,
        verify_attestation=_verify_packed,
        extension_policy=policy,
    )


class _Authenticator:
    """Minimal software authenticator producing packed self-attestations."""

    def __init__(self):
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.public_key = self.private_key.public_key()

    def make_registration(
        self,
        challenge,
        authenticator_extensions=None,
        client_extension_results=None,
        tamper_extension_bytes=False,
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
            b"\0" * 16,
            CRED_ID,
            _cose_public_key(self.public_key),
        )

        auth_data = AuthenticatorData.create(
            RP.id_hash,
            flags,
            0,
            credential_data,
            authenticator_extensions,
        )
        sig = self.private_key.sign(
            bytes(auth_data) + client_data.hash, ec.ECDSA(hashes.SHA256())
        )
        att_obj = AttestationObject.create(
            "packed",
            auth_data,
            {"alg": -7, "sig": sig},
        )

        if tamper_extension_bytes:
            # Flip the CBOR-encoded credProtect level inside the signed bytes.
            raw = bytearray(att_obj)
            decoded = cbor.decode(raw)
            ad = bytearray(decoded["authData"])
            level_offset = len(ad) - 1
            assert ad[level_offset] == 3
            ad[level_offset] = 2
            decoded["authData"] = bytes(ad)
            att_obj = AttestationObject(cbor.encode(decoded))

        return RegistrationResponse(
            raw_id=CRED_ID,
            response=AuthenticatorAttestationResponse(
                client_data=client_data,
                attestation_object=att_obj,
            ),
            client_extension_results=client_extension_results or {},
        )


def _cose_public_key(public_key):
    from fido2.cose import ES256

    return ES256.from_cryptography_key(public_key)


class TestCredProtectPolicy(unittest.TestCase):
    def setUp(self):
        self.authenticator = _Authenticator()
        self.policy = CredProtectPolicy(3)
        self.server = _make_server(self.policy)
        self.extensions = {
            "credentialProtectionPolicy": "userVerificationRequired",
            "enforceCredentialProtectionPolicy": True,
        }

    def _begin(self, policy=None):
        return self.server.register_begin(
            {"id": b"user_id", "name": "A. User"},
            extensions=self.extensions,
            extension_policy=policy,
        )

    def test_attested_credprotect_level_3_accepted(self):
        options, state = self._begin()
        self.assertEqual(options["publicKey"]["extensions"], self.extensions)
        response = self.authenticator.make_registration(
            options["publicKey"]["challenge"],
            authenticator_extensions={"credProtect": 3},
        )
        auth_data = self.server.register_complete(state, response)
        self.assertEqual(auth_data.extensions, {"credProtect": 3})

    def test_client_only_claim_is_rejected(self):
        options, state = self._begin()
        response = self.authenticator.make_registration(
            options["publicKey"]["challenge"],
            authenticator_extensions=None,
            # The browser claims the policy was applied, but nothing in the
            # signed authenticator data attests it.
            client_extension_results={"credProtect": 3},
        )
        with self.assertRaises(MissingExtensionError) as ctx:
            self.server.register_complete(state, response)
        self.assertEqual(ctx.exception.extension, "credProtect")

    def test_tampered_signed_extension_fails_attestation_first(self):
        options, state = self._begin()
        response = self.authenticator.make_registration(
            options["publicKey"]["challenge"],
            authenticator_extensions={"credProtect": 3},
            tamper_extension_bytes=True,
        )
        # The altered bytes break the attestation signature before the policy
        # is ever consulted: the evidence itself is no longer trustworthy.
        with self.assertRaises(InvalidSignature):
            self.server.register_complete(state, response)

    def test_lower_attested_level_rejected(self):
        options, state = self._begin()
        response = self.authenticator.make_registration(
            options["publicKey"]["challenge"],
            authenticator_extensions={"credProtect": 2},
        )
        with self.assertRaises(InsufficientExtensionLevelError) as ctx:
            self.server.register_complete(state, response)
        self.assertEqual(ctx.exception.required, 3)
        self.assertEqual(ctx.exception.actual, 2)

    def test_wrong_value_type_rejected(self):
        options, state = self._begin()
        response = self.authenticator.make_registration(
            options["publicKey"]["challenge"],
            authenticator_extensions={"credProtect": "3"},
        )
        with self.assertRaises(InvalidExtensionValueError):
            self.server.register_complete(state, response)

    def test_boolean_value_rejected(self):
        options, state = self._begin()
        response = self.authenticator.make_registration(
            options["publicKey"]["challenge"],
            authenticator_extensions={"credProtect": True},
        )
        with self.assertRaises(InvalidExtensionValueError):
            self.server.register_complete(state, response)

    def test_policy_sees_client_and_authenticator_extensions_separately(self):
        seen = {}

        class RecordingPolicy(AbstractExtensionPolicy):
            def check_registration_extensions(self, evidence):
                seen["evidence"] = evidence

        options, state = self._begin(policy=RecordingPolicy())
        response = self.authenticator.make_registration(
            options["publicKey"]["challenge"],
            authenticator_extensions={"credProtect": 3, "unknownExt": {"k": 1}},
            client_extension_results={"credProps": {"rk": True}},
        )
        self.server.register_complete(state, response)

        evidence = seen["evidence"]
        self.assertIsInstance(evidence, RegistrationExtensionEvidence)
        self.assertEqual(dict(evidence.requested_extensions), self.extensions)
        self.assertEqual(evidence.authenticator_extensions["credProtect"], 3)
        # Unknown, non-required extensions are preserved as decoded values.
        self.assertEqual(evidence.authenticator_extensions["unknownExt"], {"k": 1})
        # Client reported values stay in their own, unsigned bucket.
        self.assertEqual(evidence.client_extensions["credProps"], {"rk": True})
        self.assertNotIn("credProps", evidence.authenticator_extensions)

    def test_evidence_is_read_only(self):
        test_case = self
        captured = []

        class MutatingPolicy(AbstractExtensionPolicy):
            def check_registration_extensions(self, evidence):
                captured.append(evidence)
                with test_case.assertRaises(TypeError):
                    evidence.authenticator_extensions["credProtect"] = 1
                with test_case.assertRaises(TypeError):
                    evidence.authenticator_extensions["unknownExt"]["k"] = 2
                with test_case.assertRaises(TypeError):
                    evidence.client_extensions["x"] = 1
                with test_case.assertRaises(TypeError):
                    evidence.requested_extensions["x"] = 1

        options, state = self._begin(policy=MutatingPolicy())
        response = self.authenticator.make_registration(
            options["publicKey"]["challenge"],
            authenticator_extensions={"credProtect": 3, "unknownExt": {"k": 1}},
        )
        self.server.register_complete(state, response)
        evidence = captured[0]
        self.assertEqual(evidence.authenticator_extensions["credProtect"], 3)

    def test_raw_authenticator_data_unchanged_after_policy(self):
        options, state = self._begin()
        response = self.authenticator.make_registration(
            options["publicKey"]["challenge"],
            authenticator_extensions={"credProtect": 3},
        )
        raw_before = bytes(response.response.attestation_object.auth_data)
        self.server.register_complete(state, response)
        raw_after = bytes(response.response.attestation_object.auth_data)
        self.assertEqual(raw_before, raw_after)

    def test_request_config_not_mutated(self):
        options, state = self._begin()
        response = self.authenticator.make_registration(
            options["publicKey"]["challenge"],
            authenticator_extensions={"credProtect": 3},
        )
        self.server.register_complete(state, response)
        self.assertEqual(options["publicKey"]["extensions"], self.extensions)

    def test_callable_policy_supported(self):
        def require_credprotect(evidence):
            CredProtectPolicy(3).check_registration_extensions(evidence)

        server = _make_server(require_credprotect)
        options, state = server.register_begin(
            {"id": b"user_id", "name": "A. User"},
            extensions=self.extensions,
        )
        good = self.authenticator.make_registration(
            options["publicKey"]["challenge"],
            authenticator_extensions={"credProtect": 3},
        )
        server.register_complete(state, good)

        options, state = server.register_begin(
            {"id": b"user_id", "name": "A. User"},
            extensions=self.extensions,
        )
        bad = self.authenticator.make_registration(
            options["publicKey"]["challenge"],
            authenticator_extensions=None,
        )
        with self.assertRaises(MissingExtensionError):
            server.register_complete(state, bad)

    def test_no_policy_keeps_legacy_behavior(self):
        server = Fido2Server(RP)
        options, state = server.register_begin({"id": b"user_id", "name": "A. User"})
        response = self.authenticator.make_registration(
            options["publicKey"]["challenge"],
            authenticator_extensions=None,
        )
        auth_data = server.register_complete(state, response)
        self.assertIsNone(auth_data.extensions)

    def test_invalid_policy_levels(self):
        with self.assertRaises(ValueError):
            CredProtectPolicy(0)
        with self.assertRaises(ValueError):
            CredProtectPolicy(4)
        with self.assertRaises(TypeError):
            CredProtectPolicy(True)  # type: ignore[arg-type]

    def test_per_registration_policy_overrides_server_default(self):
        # The server defaults to level 3, but this registration relaxes it.
        options, state = self._begin(policy=CredProtectPolicy(1))
        response = self.authenticator.make_registration(
            options["publicKey"]["challenge"],
            authenticator_extensions={"credProtect": 1},
        )
        auth_data = self.server.register_complete(state, response)
        self.assertEqual(auth_data.extensions, {"credProtect": 1})

        # A separate registration without an override still uses the default.
        options, state = self._begin()
        response = self.authenticator.make_registration(
            options["publicKey"]["challenge"],
            authenticator_extensions={"credProtect": 1},
        )
        with self.assertRaises(InsufficientExtensionLevelError):
            self.server.register_complete(state, response)


class TestAuthenticatorDataExtensionParsing(unittest.TestCase):
    def _auth_data(self, payload, ed_flag):
        return RP.id_hash + bytes([ed_flag]) + (0).to_bytes(4, "big") + payload

    def test_ed_flag_without_cbor_map_rejected(self):
        # ED flag set, but the trailing data is not a CBOR map.
        raw = self._auth_data(b"\x01", AuthenticatorData.FLAG.ED)
        with self.assertRaises(ValueError):
            AuthenticatorData(raw)

    def test_trailing_bytes_after_extension_map_rejected(self):
        raw = self._auth_data(
            cbor.encode({"credProtect": 3}) + b"\x00",
            AuthenticatorData.FLAG.ED,
        )
        with self.assertRaises(ValueError):
            AuthenticatorData(raw)

    def test_extension_data_without_ed_flag_rejected(self):
        raw = self._auth_data(
            cbor.encode({"credProtect": 3}),
            AuthenticatorData.FLAG.UP,
        )
        with self.assertRaises(ValueError):
            AuthenticatorData(raw)

    def test_valid_extension_map_parsed(self):
        raw = self._auth_data(
            cbor.encode({"credProtect": 3}),
            AuthenticatorData.FLAG.ED,
        )
        auth_data = AuthenticatorData(raw)
        self.assertEqual(auth_data.extensions, {"credProtect": 3})
        self.assertTrue(auth_data.has_extension_data())


class TestEvidenceContainer(unittest.TestCase):
    def test_default_empty_evidence(self):
        evidence = RegistrationExtensionEvidence(
            requested_extensions=MappingProxyType({}),
            authenticator_extensions=MappingProxyType({}),
            client_extensions=MappingProxyType({}),
        )
        self.assertEqual(dict(evidence.authenticator_extensions), {})
        with self.assertRaises(TypeError):
            evidence.authenticator_extensions["a"] = 1
