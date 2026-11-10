import hashlib
import json
import unittest

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from fido2.cose import ES256
from fido2.server import AuthenticationResult, Fido2Server
from fido2.utils import websafe_encode
from fido2.webauthn import (
    Aaguid,
    AttestedCredentialData,
    AuthenticationResponse,
    AuthenticatorAssertionResponse,
    AuthenticatorData,
    CollectedClientData,
    PublicKeyCredentialRpEntity,
    UserVerificationRequirement,
)

from .test_ctap2 import _CRED_ID

RP = PublicKeyCredentialRpEntity(name="Example", id="example.com")
ORIGIN = "https://example.com"
RP_ID_HASH = hashlib.sha256(RP.id.encode()).digest()

# Two distinct accounts, each owning its own credential and private key.
USER_A = b"user-handle-a"
USER_B = b"user-handle-b"


def _make_credential(credential_id: bytes):
    """Create an AttestedCredentialData and the matching EC private key."""
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = ES256.from_cryptography_key(private_key.public_key())
    credential = AttestedCredentialData.create(Aaguid.NONE, credential_id, public_key)
    return credential, private_key


def _sign_assertion(private_key, authenticator_data, client_data):
    return private_key.sign(
        authenticator_data + client_data.hash, ec.ECDSA(hashes.SHA256())
    )


class AuthenticationFactory:
    """Builds well-formed, correctly signed authentication responses.

    Each instance holds the challenge/state for one authentication ceremony and
    can produce assertions from either account's credential, optionally tampered
    with, which lets the tests interleave the two accounts.
    """

    def __init__(self, server: Fido2Server, challenge: bytes = b"0" * 32):
        self.server = server
        self.challenge = challenge

    def begin(self, credentials=None, user_verification=None):
        options, state = self.server.authenticate_begin(
            credentials, user_verification=user_verification, challenge=self.challenge
        )
        self.state = state
        return options, state

    def response(
        self,
        credential,
        private_key,
        *,
        user_handle=b"ignored-by-pinned-flow",
        signature=None,
        challenge=None,
        origin=ORIGIN,
        uv=False,
    ):
        client_data = CollectedClientData.create(
            CollectedClientData.TYPE.GET,
            websafe_encode(challenge if challenge is not None else self.challenge),
            origin,
        )
        flags = AuthenticatorData.FLAG.UP
        if uv:
            flags |= AuthenticatorData.FLAG.UV
        authenticator_data = AuthenticatorData.create(RP_ID_HASH, flags, 1)
        if signature is None:
            signature = _sign_assertion(private_key, authenticator_data, client_data)
        return AuthenticationResponse(
            raw_id=credential.credential_id,
            response=AuthenticatorAssertionResponse(
                client_data=client_data,
                authenticator_data=authenticator_data,
                signature=signature,
                user_handle=user_handle,
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
        server = Fido2Server(RP)
        credential, _ = _make_credential(_CRED_ID)
        factory = AuthenticationFactory(server)
        _, state = factory.begin([credential])

        client_data = CollectedClientData.create(
            CollectedClientData.TYPE.GET,
            websafe_encode(factory.challenge),
            ORIGIN,
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
            server.authenticate_complete(state, [credential], response)


class TestAuthenticationScope(unittest.TestCase):
    def setUp(self):
        self.server = Fido2Server(RP)
        self.cred_a, self.key_a = _make_credential(b"credential-id-a")
        self.cred_b, self.key_b = _make_credential(b"credential-id-b")
        # All registered credentials the server knows about, both accounts.
        self.all_credentials = [self.cred_a, self.cred_b]
        self.factory = AuthenticationFactory(self.server)

    def test_begin_pins_allowed_credential_ids_in_state(self):
        _, state = self.factory.begin([self.cred_a])

        self.assertEqual(
            state["allow_credentials"],
            (websafe_encode(self.cred_a.credential_id),),
        )
        # The state must stay JSON serializable for session storage.
        json.dumps(state)

    def test_discoverable_flow_state_has_no_pinned_scope(self):
        _, state = self.factory.begin()
        self.assertIsNone(state["allow_credentials"])

    def test_pinned_scope_authenticates_allowed_credential(self):
        _, state = self.factory.begin([self.cred_a])
        response = self.factory.response(self.cred_a, self.key_a)

        result = self.server.authenticate_complete(
            state, self.all_credentials, response
        )

        self.assertIsInstance(result, AuthenticationResult)
        self.assertEqual(result.credential, self.cred_a)

    def test_valid_signature_but_credential_outside_allowed_set_is_rejected(self):
        # The ceremony is scoped to account A's credential...
        _, state = self.factory.begin([self.cred_a])
        # ...but the response is a perfectly valid signature from account B.
        response = self.factory.response(self.cred_b, self.key_b)

        with self.assertRaisesRegex(ValueError, "not part of the credentials allowed"):
            self.server.authenticate_complete(state, self.all_credentials, response)

    def test_adding_credentials_at_complete_cannot_widen_scope(self):
        _, state = self.factory.begin([self.cred_a])
        response = self.factory.response(self.cred_b, self.key_b)

        # Even passing an explicitly enlarged candidate set does not help.
        with self.assertRaises(ValueError):
            self.server.authenticate_complete(
                state, self.all_credentials + [self.cred_b], response
            )

    def test_mutating_original_list_after_begin_cannot_widen_scope(self):
        allowed = [self.cred_a]
        _, state = self.factory.begin(allowed)
        # The caller mutates the list after the scope has been pinned.
        allowed.append(self.cred_b)
        response = self.factory.response(self.cred_b, self.key_b)

        with self.assertRaises(ValueError):
            self.server.authenticate_complete(state, self.all_credentials, response)

    def test_unknown_credential_id_rejected(self):
        unknown, unknown_key = _make_credential(b"unknown-credential")
        # The ID is in the pinned scope, but the credential store has no key for it.
        _, state = self.factory.begin([unknown])
        response = self.factory.response(unknown, unknown_key)

        with self.assertRaisesRegex(ValueError, "Unknown credential ID."):
            self.server.authenticate_complete(state, self.all_credentials, response)

    def test_duplicate_credential_id_with_conflicting_public_key_is_ambiguous(self):
        # Same credential ID as A, but a different (conflicting) key pair.
        conflicting, _ = _make_credential(self.cred_a.credential_id)
        self.assertEqual(conflicting.credential_id, self.cred_a.credential_id)
        self.assertNotEqual(conflicting.public_key, self.cred_a.public_key)

        _, state = self.factory.begin([self.cred_a])
        response = self.factory.response(self.cred_a, self.key_a)

        for ordering in (
            [self.cred_a, conflicting],
            [conflicting, self.cred_a],
        ):
            with self.assertRaisesRegex(
                ValueError, "Ambiguous credential configuration"
            ):
                self.server.authenticate_complete(state, ordering, response)

    def test_duplicate_credential_id_with_same_public_key_is_accepted(self):
        # Re-parsing the same bytes yields an identical, non-conflicting entry.
        duplicate = AttestedCredentialData(bytes(self.cred_a))
        _, state = self.factory.begin([self.cred_a])
        response = self.factory.response(self.cred_a, self.key_a)

        result = self.server.authenticate_complete(
            state, [duplicate, self.cred_a], response
        )
        self.assertEqual(result.credential.credential_id, self.cred_a.credential_id)


class TestDiscoverableUserHandle(unittest.TestCase):
    def setUp(self):
        self.server = Fido2Server(RP)
        self.cred_a, self.key_a = _make_credential(b"credential-id-a")
        self.cred_b, self.key_b = _make_credential(b"credential-id-b")
        self.all_credentials = [self.cred_a, self.cred_b]
        self.credential_users = {
            self.cred_a.credential_id: USER_A,
            self.cred_b.credential_id: USER_B,
        }
        self.factory = AuthenticationFactory(self.server)

    def test_requires_expected_user_or_trusted_mapping(self):
        _, state = self.factory.begin()
        response = self.factory.response(self.cred_a, self.key_a, user_handle=USER_A)

        with self.assertRaisesRegex(
            ValueError, "expected_user_handle or a credential_users mapping"
        ):
            self.server.authenticate_complete(state, self.all_credentials, response)

    def test_mapping_owner_matching_user_handle_authenticates(self):
        _, state = self.factory.begin()
        response = self.factory.response(self.cred_a, self.key_a, user_handle=USER_A)

        result = self.server.authenticate_complete(
            state,
            self.all_credentials,
            response,
            credential_users=self.credential_users,
        )

        self.assertEqual(result.credential, self.cred_a)
        self.assertEqual(result.user_handle, USER_A)

    def test_expected_user_handle_matching_authenticates(self):
        _, state = self.factory.begin()
        response = self.factory.response(self.cred_a, self.key_a, user_handle=USER_A)

        result = self.server.authenticate_complete(
            state,
            self.all_credentials,
            response,
            expected_user_handle=USER_A,
        )
        self.assertEqual(result.user_handle, USER_A)

    def test_missing_user_handle_is_distinct_and_rejected(self):
        _, state = self.factory.begin()
        response = self.factory.response(self.cred_a, self.key_a, user_handle=None)

        with self.assertRaisesRegex(ValueError, "Missing user handle"):
            self.server.authenticate_complete(
                state,
                self.all_credentials,
                response,
                credential_users=self.credential_users,
            )

    def test_empty_user_handle_is_distinct_from_missing_and_rejected(self):
        _, state = self.factory.begin()
        # An empty (but present) handle must not be confused with a missing one.
        response = self.factory.response(self.cred_a, self.key_a, user_handle=b"")

        with self.assertRaisesRegex(
            ValueError, "does not match the credential's owner"
        ):
            self.server.authenticate_complete(
                state,
                self.all_credentials,
                response,
                credential_users=self.credential_users,
            )

    def test_correct_credential_but_user_handle_of_other_account_rejected(self):
        # Credential A signs correctly, yet the assertion claims account B.
        _, state = self.factory.begin()
        response = self.factory.response(self.cred_a, self.key_a, user_handle=USER_B)

        with self.assertRaisesRegex(
            ValueError, "does not match the credential's owner"
        ):
            self.server.authenticate_complete(
                state,
                self.all_credentials,
                response,
                credential_users=self.credential_users,
            )

    def test_expected_user_handle_mismatch_rejected(self):
        _, state = self.factory.begin()
        response = self.factory.response(self.cred_a, self.key_a, user_handle=USER_B)

        with self.assertRaisesRegex(ValueError, "does not match the expected user"):
            self.server.authenticate_complete(
                state,
                self.all_credentials,
                response,
                expected_user_handle=USER_A,
            )

    def test_conflicting_expected_user_and_mapping_owner_rejected(self):
        _, state = self.factory.begin()
        response = self.factory.response(self.cred_a, self.key_a, user_handle=USER_A)

        with self.assertRaisesRegex(ValueError, "Ambiguous credential configuration"):
            self.server.authenticate_complete(
                state,
                self.all_credentials,
                response,
                expected_user_handle=USER_B,
                credential_users=self.credential_users,
            )

    def test_credential_absent_from_trusted_mapping_rejected(self):
        _, state = self.factory.begin()
        response = self.factory.response(self.cred_a, self.key_a, user_handle=USER_A)

        with self.assertRaisesRegex(ValueError, "not associated with a known user"):
            self.server.authenticate_complete(
                state,
                self.all_credentials,
                response,
                credential_users={self.cred_b.credential_id: USER_B},
            )


class TestAuthenticationGuards(unittest.TestCase):
    def setUp(self):
        self.server = Fido2Server(RP)
        self.cred_a, self.key_a = _make_credential(b"credential-id-a")
        self.factory = AuthenticationFactory(self.server)

    def _complete(self, state, response):
        return self.server.authenticate_complete(state, [self.cred_a], response)

    def test_wrong_challenge_rejected(self):
        _, state = self.factory.begin([self.cred_a])
        response = self.factory.response(self.cred_a, self.key_a, challenge=b"1" * 32)
        with self.assertRaisesRegex(ValueError, "Wrong challenge"):
            self._complete(state, response)

    def test_wrong_origin_rejected(self):
        _, state = self.factory.begin([self.cred_a])
        response = self.factory.response(
            self.cred_a, self.key_a, origin="https://evil.example"
        )
        with self.assertRaisesRegex(ValueError, "Invalid origin"):
            self._complete(state, response)

    def test_wrong_rp_id_hash_rejected(self):
        _, state = self.factory.begin([self.cred_a])
        client_data = CollectedClientData.create(
            CollectedClientData.TYPE.GET,
            websafe_encode(self.factory.challenge),
            ORIGIN,
        )
        bad_auth_data = AuthenticatorData.create(
            b"\x11" * 32, AuthenticatorData.FLAG.UP, 1
        )
        signature = _sign_assertion(self.key_a, bad_auth_data, client_data)
        response = AuthenticationResponse(
            raw_id=self.cred_a.credential_id,
            response=AuthenticatorAssertionResponse(
                client_data=client_data,
                authenticator_data=bad_auth_data,
                signature=signature,
            ),
        )
        with self.assertRaisesRegex(ValueError, "Wrong RP ID hash"):
            self._complete(state, response)

    def test_user_verification_required_but_not_performed_rejected(self):
        _, state = self.factory.begin(
            [self.cred_a], user_verification=UserVerificationRequirement.REQUIRED
        )
        response = self.factory.response(self.cred_a, self.key_a, uv=False)
        with self.assertRaisesRegex(ValueError, "User verification required"):
            self._complete(state, response)

    def test_user_verification_required_and_performed_authenticates(self):
        _, state = self.factory.begin(
            [self.cred_a], user_verification=UserVerificationRequirement.REQUIRED
        )
        response = self.factory.response(self.cred_a, self.key_a, uv=True)
        result = self._complete(state, response)
        self.assertEqual(result.credential, self.cred_a)

    def test_invalid_signature_rejected_after_scope_checks(self):
        _, state = self.factory.begin([self.cred_a])
        response = self.factory.response(self.cred_a, self.key_a, signature=b"INVALID")
        with self.assertRaisesRegex(ValueError, "Invalid signature."):
            self._complete(state, response)
