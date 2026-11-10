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

"""Binding-oriented negative tests for attestation verification.

tests/test_attestation.py verifies that genuine, captured attestation
samples are accepted. This file complements it with locally generated
samples (a local CA, a fixed short client message, no authenticator and
no network access) designed to catch *missing field bindings*: each
counterexample starts from a valid sample and replaces exactly one
relevant object. Where possible the outer signature is re-made valid,
so the sample can only be rejected by an internal binding check and not
by signature verification alone.

Note on scope: python-fido2 implements the Android attestation format
"android-safetynet" (fido2/attestation/android.py); there is no
"android-key" implementation in this version, so the Android coverage
below targets android-safetynet.

Failure categories
------------------
1. Parse failure: malformed structures -> InvalidData.
2. Signature failure: signed data was replaced without re-signing
   -> InvalidSignature.
3. Trust path failure: the certificate chain does not verify
   (verify_x509_chain raises InvalidSignature), or the attestation
   certificate violates format requirements -> InvalidData.
4. Binding failure: the outer signature is valid, but internal evidence
   (nonce, extraData, AAGUID extension, pubArea, ...) does not match
   the registration request -> InvalidData or InvalidSignature.

Every counterexample docstring records what was changed, why the outer
signature is still valid, and which internal check is expected to catch
it, so the corpus can be extended following the same pattern.

One note on packed: per spec, the credential public key in a "packed"
basic attestation is bound only by the signature over authData (the
attestation certificate does not certify the credential key). Swapping
the credential key and re-signing therefore yields a *valid* sample and
is not a counterexample; the binding is tested by swapping the key
*without* re-signing (category 2).
"""

import importlib.util
import json
import struct
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.exceptions import InvalidSignature as CryptoInvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from fido2.attestation import (
    AndroidSafetynetAttestation,
    AttestationType,
    InvalidData,
    InvalidSignature,
    PackedAttestation,
    TpmAttestation,
    verify_x509_chain,
)
from fido2.attestation.packed import OID_AAGUID
from fido2.attestation.tpm import OID_AIK_CERTIFICATE
from fido2.cose import ES256, RS256
from fido2.utils import sha256, websafe_encode
from fido2.webauthn import AttestedCredentialData, AuthenticatorData

# Fixed short messages: the tests never depend on wall-clock, network or
# authenticator state.
CLIENT_DATA_HASH = sha256(b"python-fido2 attestation binding test")
OTHER_CLIENT_DATA_HASH = sha256(b"a different client message")
RP_ID_HASH = sha256(b"example.com")
OTHER_RP_ID_HASH = sha256(b"other.example.com")
AAGUID = bytes(range(16))
OTHER_AAGUID = bytes(range(16, 32))
CREDENTIAL_ID = b"binding-test-credential"

_NOT_BEFORE = datetime(2026, 1, 1)
_NOT_AFTER = datetime(2027, 1, 1)


def _der(cert):
    return cert.public_bytes(serialization.Encoding.DER)


def _sign(key, message):
    if isinstance(key, rsa.RSAPrivateKey):
        return key.sign(message, padding.PKCS1v15(), hashes.SHA256())
    return key.sign(message, ec.ECDSA(hashes.SHA256()))


def _issue_leaf(public_key, ca, subject, extra_extensions=()):
    ca_key, ca_cert = ca
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(public_key)
        .serial_number(2)
        .not_valid_before(_NOT_BEFORE)
        .not_valid_after(_NOT_AFTER)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), True)
    )
    for value, critical in extra_extensions:
        builder = builder.add_extension(value, critical)
    return builder.sign(ca_key, hashes.SHA256())


def _auth_data(cose_key, aaguid=AAGUID, rp_id_hash=RP_ID_HASH):
    credential_data = AttestedCredentialData.create(aaguid, CREDENTIAL_ID, cose_key)
    return AuthenticatorData.create(
        rp_id_hash,
        AuthenticatorData.FLAG.UP | AuthenticatorData.FLAG.AT,
        0,
        bytes(credential_data),
    )


@pytest.fixture(scope="module")
def ca():
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, "Binding CA")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(_NOT_BEFORE)
        .not_valid_after(_NOT_AFTER)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), True)
        .sign(key, hashes.SHA256())
    )
    return key, cert


@pytest.fixture(scope="module")
def other_ca():
    # A second CA reusing the same subject name but a different key, so
    # that chain verification fails on the signature, not the issuer name.
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, "Binding CA")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(_NOT_BEFORE)
        .not_valid_after(_NOT_AFTER)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), True)
        .sign(key, hashes.SHA256())
    )
    return key, cert


@pytest.fixture(scope="module")
def credential_key():
    return ec.generate_private_key(ec.SECP256R1())


@pytest.fixture(scope="module")
def credential_cose(credential_key):
    return ES256.from_cryptography_key(credential_key.public_key())


@pytest.fixture(scope="module")
def auth_data(credential_cose):
    return _auth_data(credential_cose)


# --- packed -----------------------------------------------------------------

PACKED_SUBJECT = x509.Name(
    [
        x509.NameAttribute(x509.NameOID.COUNTRY_NAME, "SE"),
        x509.NameAttribute(x509.NameOID.ORGANIZATION_NAME, "Example Corp"),
        x509.NameAttribute(
            x509.NameOID.ORGANIZATIONAL_UNIT_NAME, "Authenticator Attestation"
        ),
        x509.NameAttribute(x509.NameOID.COMMON_NAME, "Packed Binding Test"),
    ]
)


def _packed_cert(key, ca, aaguid=AAGUID, subject=PACKED_SUBJECT):
    extensions = []
    if aaguid is not None:
        # The AAGUID extension carries a DER OCTET STRING wrapping the AAGUID.
        extensions.append(
            (x509.UnrecognizedExtension(OID_AAGUID, b"\x04\x10" + aaguid), False)
        )
    return _issue_leaf(key.public_key(), ca, subject, extensions)


@pytest.fixture(scope="module")
def packed_ec(ca):
    key = ec.generate_private_key(ec.SECP256R1())
    return key, -7, _packed_cert(key, ca)


@pytest.fixture(scope="module")
def packed_rsa(ca):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key, -257, _packed_cert(key, ca)


def _packed_statement(attester, ca, auth_data, client_data_hash=CLIENT_DATA_HASH):
    key, alg, cert = attester
    return {
        "alg": alg,
        "sig": _sign(key, bytes(auth_data) + client_data_hash),
        "x5c": [_der(cert), _der(ca[1])],
    }


class TestPackedBinding:
    def test_valid_algorithm_combinations(self, packed_ec, packed_rsa, ca, auth_data):
        """Control: ES256 and RS256 attestation certificates both verify."""
        for attester in (packed_ec, packed_rsa):
            statement = _packed_statement(attester, ca, auth_data)
            result = PackedAttestation().verify(statement, auth_data, CLIENT_DATA_HASH)
            assert result.attestation_type == AttestationType.BASIC
            verify_x509_chain(result.trust_path)

    def test_aaguid_extension_optional(self, ca, credential_cose, auth_data):
        """Control: the AAGUID certificate extension is optional."""
        key = ec.generate_private_key(ec.SECP256R1())
        cert = _packed_cert(key, ca, aaguid=None)
        statement = _packed_statement((key, -7, cert), ca, auth_data)
        result = PackedAttestation().verify(statement, auth_data, CLIENT_DATA_HASH)
        assert result.attestation_type == AttestationType.BASIC

    def test_self_attestation_valid(self, credential_key, credential_cose, auth_data):
        """Control: self attestation (no x5c) verifies."""
        statement = {
            "alg": -7,
            "sig": _sign(credential_key, bytes(auth_data) + CLIENT_DATA_HASH),
        }
        result = PackedAttestation().verify(statement, auth_data, CLIENT_DATA_HASH)
        assert result.attestation_type == AttestationType.SELF
        assert result.trust_path == []

    def test_client_data_hash_not_signed(self, packed_ec, ca, auth_data):
        """Category 2 (signature failure).

        Changed: clientDataHash is replaced at verification time.
        Outer signature: stale (still covers the original clientDataHash),
        which is exactly what binds clientDataHash in packed.
        Caught by: signature verification over authData || clientDataHash.
        """
        statement = _packed_statement(packed_ec, ca, auth_data)
        with pytest.raises(InvalidSignature):
            PackedAttestation().verify(statement, auth_data, OTHER_CLIENT_DATA_HASH)

    def test_auth_data_replaced(self, packed_ec, ca, credential_cose):
        """Category 2 (signature failure).

        Changed: authenticatorData (rpIdHash) is replaced after signing.
        Outer signature: stale, binding authenticatorData in packed.
        Caught by: signature verification over authData || clientDataHash.
        """
        statement = _packed_statement(
            packed_ec, ca, _auth_data(credential_cose, rp_id_hash=RP_ID_HASH)
        )
        replaced = _auth_data(credential_cose, rp_id_hash=OTHER_RP_ID_HASH)
        with pytest.raises(InvalidSignature):
            PackedAttestation().verify(statement, replaced, CLIENT_DATA_HASH)

    def test_credential_key_replaced(self, packed_ec, ca, auth_data):
        """Category 2 (signature failure).

        Changed: the credential public key inside authenticatorData.
        Outer signature: stale. In packed, the credential key is bound
        only by the signature over authData, so replacing it without
        re-signing must fail signature verification.
        Caught by: signature verification over authData || clientDataHash.
        """
        statement = _packed_statement(packed_ec, ca, auth_data)
        other_key = ec.generate_private_key(ec.SECP256R1())
        replaced = _auth_data(ES256.from_cryptography_key(other_key.public_key()))
        with pytest.raises(InvalidSignature):
            PackedAttestation().verify(statement, replaced, CLIENT_DATA_HASH)

    def test_aaguid_mismatch_resigned(self, packed_ec, ca, credential_cose):
        """Category 4 (binding failure).

        Changed: AAGUID in authenticatorData no longer matches the
        AAGUID extension of the attestation certificate.
        Outer signature: re-signed over the modified authData with the
        attestation key, so it is valid; only the certificate binding
        can reject this sample.
        Caught by: _validate_packed_cert AAGUID extension comparison.
        """
        key, alg, cert = packed_ec  # certificate carries AAGUID
        replaced = _auth_data(credential_cose, aaguid=OTHER_AAGUID)
        statement = _packed_statement((key, alg, cert), ca, replaced)
        with pytest.raises(InvalidData, match="AAGUID"):
            PackedAttestation().verify(statement, replaced, CLIENT_DATA_HASH)

    def test_self_alg_mismatch(self, credential_key, credential_cose, auth_data):
        """Category 4 (binding failure).

        Changed: statement "alg" claims ES384 while the credential key
        is ES256.
        Outer signature: validly made with the credential key; the alg
        claim is checked before (and independently of) it.
        Caught by: "Wrong algorithm of public key!" check.
        """
        statement = {
            "alg": -35,  # ES384
            "sig": _sign(credential_key, bytes(auth_data) + CLIENT_DATA_HASH),
        }
        with pytest.raises(InvalidData, match="Wrong algorithm"):
            PackedAttestation().verify(statement, auth_data, CLIENT_DATA_HASH)

    def test_malformed_certificate(self, packed_ec, ca, auth_data):
        """Category 1 (parse failure).

        Changed: x5c[0] is not a DER certificate.
        Caught by: certificate parsing -> InvalidData.
        """
        statement = _packed_statement(packed_ec, ca, auth_data)
        statement["x5c"] = [b"not a certificate"]
        with pytest.raises(InvalidData):
            PackedAttestation().verify(statement, auth_data, CLIENT_DATA_HASH)

    def test_broken_signature(self, packed_ec, ca, auth_data):
        """Category 2 (signature failure): baseline corrupted signature."""
        statement = _packed_statement(packed_ec, ca, auth_data)
        statement["sig"] = b"\x00" * len(statement["sig"])
        with pytest.raises(InvalidSignature):
            PackedAttestation().verify(statement, auth_data, CLIENT_DATA_HASH)

    def test_untrusted_chain(self, packed_ec, ca, other_ca, auth_data):
        """Category 3 (trust path failure).

        Changed: the chain root is replaced by an unrelated CA.
        Caught by: verify_x509_chain on the returned trust path.
        """
        statement = _packed_statement(packed_ec, ca, auth_data)
        result = PackedAttestation().verify(statement, auth_data, CLIENT_DATA_HASH)
        with pytest.raises(InvalidSignature):
            verify_x509_chain([result.trust_path[0], _der(other_ca[1])])


# --- tpm --------------------------------------------------------------------

# TPMA_OBJECT: FIXED_TPM | FIXED_PARENT | SENSITIVE_DATA_ORIGIN |
#              USER_WITH_AUTH | SIGN_ENCRYPT
_TPM_OBJECT_ATTRIBUTES = 0x00040072
_TPM_ALG_RSA = 0x0001
_TPM_ALG_ECC = 0x0023
_TPM_ALG_SHA256 = 0x000B
_TPM_ALG_NULL = 0x0010
_TPM_ECC_CURVE_NIST_P256 = 0x0003


def _tpm2b(data):
    return struct.pack("!H", len(data)) + data


def _ecc_pub_area(public_key):
    numbers = public_key.public_numbers()
    return (
        struct.pack("!HHI", _TPM_ALG_ECC, _TPM_ALG_SHA256, _TPM_OBJECT_ATTRIBUTES)
        + _tpm2b(b"")  # authPolicy
        + struct.pack(
            "!HHHH", _TPM_ALG_NULL, _TPM_ALG_NULL, _TPM_ECC_CURVE_NIST_P256,
            _TPM_ALG_NULL,
        )
        + _tpm2b(numbers.x.to_bytes(32, "big"))
        + _tpm2b(numbers.y.to_bytes(32, "big"))
    )


def _rsa_pub_area(public_key):
    numbers = public_key.public_numbers()
    return (
        struct.pack("!HHI", _TPM_ALG_RSA, _TPM_ALG_SHA256, _TPM_OBJECT_ATTRIBUTES)
        + _tpm2b(b"")  # authPolicy
        + struct.pack("!HHHL", _TPM_ALG_NULL, _TPM_ALG_NULL, 2048, 0)
        + _tpm2b(numbers.n.to_bytes(256, "big"))
    )


def _pub_area_name(pub_area):
    return struct.pack("!H", _TPM_ALG_SHA256) + sha256(pub_area)


def _cert_info(extra_data, pub_area_name):
    return (
        b"\xffTCG"  # TPM_GENERATED_VALUE
        + b"\x80\x17"  # TPM_ST_ATTEST_CERTIFY
        + _tpm2b(struct.pack("!H", _TPM_ALG_SHA256) + b"\x00" * 32)  # signer name
        + _tpm2b(extra_data)
        + struct.pack("!QLLB", 0, 0, 0, 1)  # TPMS_CLOCK_INFO
        + struct.pack("!Q", 0)  # firmwareVersion
        + _tpm2b(pub_area_name)  # attested.certify.name
        + _tpm2b(struct.pack("!H", _TPM_ALG_SHA256) + b"\x01" * 32)  # qualified
    )


def _tpm_cert(key, ca, subject=None, aik_eku=True):
    eku = OID_AIK_CERTIFICATE if aik_eku else x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH
    return _issue_leaf(
        key.public_key(),
        ca,
        subject if subject is not None else x509.Name([]),
        [
            (x509.SubjectAlternativeName([x509.DNSName("aik.binding.test")]), False),
            (x509.ExtendedKeyUsage([eku]), False),
        ],
    )


@pytest.fixture(scope="module", params=["rsa", "ec"], ids=["RS256", "ES256"])
def aik(request, ca):
    if request.param == "rsa":
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        alg = -257
    else:
        key = ec.generate_private_key(ec.SECP256R1())
        alg = -7
    return key, alg, _tpm_cert(key, ca)


def _tpm_statement(
    aik,
    ca,
    auth_data,
    pub_area,
    client_data_hash=CLIENT_DATA_HASH,
    extra_data=None,
    pub_area_name=None,
    cert_info=None,
):
    key, alg, cert = aik
    if cert_info is None:
        if extra_data is None:
            extra_data = sha256(bytes(auth_data) + client_data_hash)
        if pub_area_name is None:
            pub_area_name = _pub_area_name(pub_area)
        cert_info = _cert_info(extra_data, pub_area_name)
    return {
        "alg": alg,
        "sig": _sign(key, cert_info),
        "x5c": [_der(cert), _der(ca[1])],
        "certInfo": cert_info,
        "pubArea": pub_area,
    }


class TestTpmBinding:
    def test_valid(self, aik, ca, credential_key, auth_data):
        """Control: RS256 and ES256 AIK certificates both verify."""
        pub_area = _ecc_pub_area(credential_key.public_key())
        statement = _tpm_statement(aik, ca, auth_data, pub_area)
        result = TpmAttestation().verify(statement, auth_data, CLIENT_DATA_HASH)
        assert result.attestation_type == AttestationType.ATT_CA
        verify_x509_chain(result.trust_path)

    def test_valid_rsa_credential(self, aik, ca):
        """Control: an RSA credential key in pubArea also verifies."""
        cred_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cose = RS256.from_cryptography_key(cred_key.public_key())
        auth_data = _auth_data(cose)
        pub_area = _rsa_pub_area(cred_key.public_key())
        statement = _tpm_statement(aik, ca, auth_data, pub_area)
        result = TpmAttestation().verify(statement, auth_data, CLIENT_DATA_HASH)
        assert result.attestation_type == AttestationType.ATT_CA

    def test_extra_data_mismatch_resigned(self, aik, ca, credential_key, auth_data):
        """Category 4 (binding failure).

        Changed: certInfo.extraData (the TPM-specific digest of
        authData || clientDataHash) commits to a different request.
        Outer signature: certInfo is re-signed with the AIK key, so the
        signature over certInfo is valid; only the extraData comparison
        can reject this sample.
        Caught by: "attestation does not sign for authData and
        ClientData" (raised as InvalidSignature).
        """
        pub_area = _ecc_pub_area(credential_key.public_key())
        statement = _tpm_statement(
            aik, ca, auth_data, pub_area, extra_data=sha256(b"another request")
        )
        with pytest.raises(InvalidSignature, match="authData and ClientData"):
            TpmAttestation().verify(statement, auth_data, CLIENT_DATA_HASH)

    def test_client_data_hash_not_bound(self, aik, ca, credential_key, auth_data):
        """Category 4 (binding failure).

        Changed: clientDataHash is replaced at verification time.
        Outer signature: untouched and valid over certInfo; in TPM the
        clientDataHash binding lives in certInfo.extraData, not in the
        signature input itself.
        Caught by: extraData comparison ("authData and ClientData").
        """
        pub_area = _ecc_pub_area(credential_key.public_key())
        statement = _tpm_statement(aik, ca, auth_data, pub_area)
        with pytest.raises(InvalidSignature, match="authData and ClientData"):
            TpmAttestation().verify(statement, auth_data, OTHER_CLIENT_DATA_HASH)

    def test_pubarea_credential_key_mismatch(self, aik, ca, credential_key, auth_data):
        """Category 4 (binding failure).

        Changed: pubArea encodes a different public key than the
        credentialPublicKey in authenticatorData. certInfo is fully
        consistent with the wrong pubArea (name and extraData match)
        and is re-signed, so the outer signature is valid.
        Caught by: pubArea vs attestedCredentialData comparison
        (raised as InvalidSignature).
        """
        other_key = ec.generate_private_key(ec.SECP256R1())
        wrong_pub_area = _ecc_pub_area(other_key.public_key())
        statement = _tpm_statement(aik, ca, auth_data, wrong_pub_area)
        with pytest.raises(InvalidSignature, match="pubArea does not match"):
            TpmAttestation().verify(statement, auth_data, CLIENT_DATA_HASH)

    def test_attested_name_mismatch_resigned(
        self, aik, ca, credential_key, auth_data
    ):
        """Category 4 (binding failure).

        Changed: certInfo.attested.certify.name is the Name of a
        different pubArea than the one shipped in the statement.
        Outer signature: certInfo is re-signed with the AIK key and is
        valid; only the name-vs-pubArea check can reject this sample.
        Caught by: "TPMS_CERTIFY_INFO does not include a valid name".
        """
        pub_area = _ecc_pub_area(credential_key.public_key())
        other_name = _pub_area_name(
            _ecc_pub_area(ec.generate_private_key(ec.SECP256R1()).public_key())
        )
        statement = _tpm_statement(
            aik, ca, auth_data, pub_area, pub_area_name=other_name
        )
        with pytest.raises(InvalidData, match="TPMS_CERTIFY_INFO"):
            TpmAttestation().verify(statement, auth_data, CLIENT_DATA_HASH)

    def test_signature_tampered(self, aik, ca, credential_key, auth_data):
        """Category 2 (signature failure): baseline corrupted signature."""
        pub_area = _ecc_pub_area(credential_key.public_key())
        statement = _tpm_statement(aik, ca, auth_data, pub_area)
        statement["sig"] = b"\x00" * len(statement["sig"])
        with pytest.raises(InvalidSignature, match="signature of certInfo"):
            TpmAttestation().verify(statement, auth_data, CLIENT_DATA_HASH)

    def test_cert_info_bad_magic_resigned(self, aik, ca, credential_key, auth_data):
        """Category 1 (parse failure).

        Changed: certInfo magic is not TPM_GENERATED_VALUE.
        Outer signature: the corrupted certInfo is re-signed and valid;
        the sample is rejected while parsing, before signature checks.
        Caught by: TpmAttestationFormat.parse -> InvalidData.
        """
        pub_area = _ecc_pub_area(credential_key.public_key())
        cert_info = _cert_info(
            sha256(bytes(auth_data) + CLIENT_DATA_HASH), _pub_area_name(pub_area)
        )
        cert_info = b"\xde\xad\xbe\xef" + cert_info[4:]
        statement = _tpm_statement(aik, ca, auth_data, pub_area, cert_info=cert_info)
        with pytest.raises(InvalidData):
            TpmAttestation().verify(statement, auth_data, CLIENT_DATA_HASH)

    def test_pubarea_malformed(self, aik, ca, auth_data):
        """Category 1 (parse failure).

        Changed: pubArea is truncated and cannot be parsed.
        Caught by: TpmPublicFormat.parse -> InvalidData.
        """
        statement = _tpm_statement(aik, ca, auth_data, b"\x00\x23")
        with pytest.raises(InvalidData, match="unable to parse pubArea"):
            TpmAttestation().verify(statement, auth_data, CLIENT_DATA_HASH)

    def test_cert_subject_not_empty(self, aik, ca, credential_key, auth_data):
        """Category 3 (trust path failure).

        Changed: the AIK certificate has a non-empty Subject, which the
        TPM certificate requirements forbid.
        Caught by: _validate_tpm_cert -> InvalidData.
        """
        key, alg, _ = aik
        cert = _tpm_cert(
            key,
            ca,
            subject=x509.Name(
                [x509.NameAttribute(x509.NameOID.COMMON_NAME, "not empty")]
            ),
        )
        pub_area = _ecc_pub_area(credential_key.public_key())
        statement = _tpm_statement((key, alg, cert), ca, auth_data, pub_area)
        with pytest.raises(InvalidData, match="should not have Subject"):
            TpmAttestation().verify(statement, auth_data, CLIENT_DATA_HASH)

    def test_cert_eku_missing_aik(self, aik, ca, credential_key, auth_data):
        """Category 3 (trust path failure).

        Changed: the AIK certificate EKU does not contain the
        tcg-kp-AIKCertificate OID.
        Caught by: _validate_tpm_cert -> InvalidData.
        """
        key, alg, _ = aik
        cert = _tpm_cert(key, ca, aik_eku=False)
        pub_area = _ecc_pub_area(credential_key.public_key())
        statement = _tpm_statement((key, alg, cert), ca, auth_data, pub_area)
        with pytest.raises(InvalidData, match="Extended key usage"):
            TpmAttestation().verify(statement, auth_data, CLIENT_DATA_HASH)

    def test_untrusted_chain(self, aik, ca, other_ca, credential_key, auth_data):
        """Category 3 (trust path failure).

        Changed: the chain root is replaced by an unrelated CA.
        Caught by: verify_x509_chain on the returned trust path.
        """
        pub_area = _ecc_pub_area(credential_key.public_key())
        statement = _tpm_statement(aik, ca, auth_data, pub_area)
        result = TpmAttestation().verify(statement, auth_data, CLIENT_DATA_HASH)
        with pytest.raises(InvalidSignature):
            verify_x509_chain([result.trust_path[0], _der(other_ca[1])])


# --- android-safetynet -------------------------------------------------------

SAFETYNET_SUBJECT = x509.Name(
    [x509.NameAttribute(x509.NameOID.COMMON_NAME, "attest.android.com")]
)


@pytest.fixture(scope="module")
def safetynet_leaf(ca):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key, _issue_leaf(key.public_key(), ca, SAFETYNET_SUBJECT)


def _safetynet_statement(
    leaf,
    ca,
    auth_data,
    client_data_hash=CLIENT_DATA_HASH,
    nonce=None,
    cts_profile_match=True,
):
    key, cert = leaf
    if nonce is None:
        nonce = sha256(bytes(auth_data) + client_data_hash)
    header = {
        "alg": "RS256",
        "x5c": [websafe_encode(_der(cert)), websafe_encode(_der(ca[1]))],
    }
    payload = {
        "nonce": websafe_encode(nonce),
        "ctsProfileMatch": cts_profile_match,
        "timestampMs": 1762732800000,
        "apkPackageName": "com.example.binding",
        "apkDigestSha256": websafe_encode(sha256(b"apk")),
        "apkCertificateDigestSha256": [websafe_encode(sha256(b"apk-cert"))],
        "basicIntegrity": True,
    }
    signing_input = (
        websafe_encode(json.dumps(header).encode()).encode()
        + b"."
        + websafe_encode(json.dumps(payload).encode()).encode()
    )
    signature = _sign(key, signing_input)
    return {
        "ver": "binding-test",
        "response": signing_input + b"." + websafe_encode(signature).encode(),
    }


class TestAndroidSafetynetBinding:
    def test_valid(self, safetynet_leaf, ca, auth_data):
        """Control: a well-formed SafetyNet response verifies."""
        statement = _safetynet_statement(safetynet_leaf, ca, auth_data)
        result = AndroidSafetynetAttestation().verify(
            statement, auth_data, CLIENT_DATA_HASH
        )
        assert result.attestation_type == AttestationType.BASIC
        verify_x509_chain(result.trust_path)

    def test_nonce_mismatch_resigned(self, safetynet_leaf, ca, auth_data):
        """Category 4 (binding failure).

        Changed: the JWT nonce (the SafetyNet-specific digest of
        authData || clientDataHash) commits to a different request.
        Outer signature: the JWT is re-signed with the leaf key, so the
        JWS signature is valid; only the nonce comparison can reject
        this sample.
        Caught by: "Nonce does not match!".
        """
        statement = _safetynet_statement(
            safetynet_leaf, ca, auth_data, nonce=sha256(b"another request")
        )
        with pytest.raises(InvalidData, match="Nonce does not match"):
            AndroidSafetynetAttestation().verify(
                statement, auth_data, CLIENT_DATA_HASH
            )

    def test_client_data_hash_not_bound(self, safetynet_leaf, ca, auth_data):
        """Category 4 (binding failure).

        Changed: clientDataHash is replaced at verification time.
        Outer signature: untouched and valid; the clientDataHash
        binding lives in the JWT nonce, not in the signature input.
        Caught by: "Nonce does not match!".
        """
        statement = _safetynet_statement(safetynet_leaf, ca, auth_data)
        with pytest.raises(InvalidData, match="Nonce does not match"):
            AndroidSafetynetAttestation().verify(
                statement, auth_data, OTHER_CLIENT_DATA_HASH
            )

    def test_auth_data_not_bound(self, safetynet_leaf, ca, credential_cose):
        """Category 4 (binding failure).

        Changed: authenticatorData is replaced at verification time
        (different rpIdHash).
        Outer signature: untouched and valid; the authenticatorData
        binding lives in the JWT nonce.
        Caught by: "Nonce does not match!".
        """
        statement = _safetynet_statement(
            safetynet_leaf, ca, _auth_data(credential_cose)
        )
        replaced = _auth_data(credential_cose, rp_id_hash=OTHER_RP_ID_HASH)
        with pytest.raises(InvalidData, match="Nonce does not match"):
            AndroidSafetynetAttestation().verify(
                statement, replaced, CLIENT_DATA_HASH
            )

    def test_cts_profile_match_false_resigned(self, safetynet_leaf, ca, auth_data):
        """Category 4 (binding failure).

        Changed: ctsProfileMatch is false (device integrity not met).
        Outer signature: the JWT is re-signed and valid.
        Caught by: "ctsProfileMatch must be true!".
        """
        statement = _safetynet_statement(
            safetynet_leaf, ca, auth_data, cts_profile_match=False
        )
        with pytest.raises(InvalidData, match="ctsProfileMatch"):
            AndroidSafetynetAttestation().verify(
                statement, auth_data, CLIENT_DATA_HASH
            )

    def test_signature_tampered(self, safetynet_leaf, ca, auth_data):
        """Category 2 (signature failure).

        Changed: the JWS signature is corrupted. Note: android.py does
        not wrap the cryptography exception, unlike packed/tpm.
        """
        statement = _safetynet_statement(safetynet_leaf, ca, auth_data)
        signing_input = statement["response"].rsplit(b".", 1)[0]
        statement["response"] = (
            signing_input + b"." + websafe_encode(b"\x00" * 256).encode()
        )
        with pytest.raises(CryptoInvalidSignature):
            AndroidSafetynetAttestation().verify(
                statement, auth_data, CLIENT_DATA_HASH
            )

    def test_malformed_response(self, auth_data):
        """Category 1 (parse failure).

        Changed: the response is not a three-part JWS.
        Caught by: JWS splitting -> InvalidData.
        """
        with pytest.raises(InvalidData):
            AndroidSafetynetAttestation().verify(
                {"ver": "1", "response": b"not-a-jwt"}, auth_data, CLIENT_DATA_HASH
            )

    def test_wrong_certificate_cn(self, safetynet_leaf, ca, auth_data):
        """Category 3 (trust path failure).

        Changed: the leaf certificate is issued to a host other than
        attest.android.com. The JWT is re-signed with that leaf key, so
        the JWS signature is valid.
        Caught by: "Certificate not issued to attest.android.com!".
        """
        key, _ = safetynet_leaf
        cert = _issue_leaf(
            key.public_key(),
            ca,
            x509.Name(
                [x509.NameAttribute(x509.NameOID.COMMON_NAME, "evil.example.com")]
            ),
        )
        statement = _safetynet_statement((key, cert), ca, auth_data)
        with pytest.raises(InvalidData, match="attest.android.com"):
            AndroidSafetynetAttestation().verify(
                statement, auth_data, CLIENT_DATA_HASH
            )

    def test_untrusted_chain(self, safetynet_leaf, ca, other_ca, auth_data):
        """Category 3 (trust path failure).

        Changed: the chain root is replaced by an unrelated CA.
        Caught by: verify_x509_chain on the returned trust path.
        """
        statement = _safetynet_statement(safetynet_leaf, ca, auth_data)
        result = AndroidSafetynetAttestation().verify(
            statement, auth_data, CLIENT_DATA_HASH
        )
        with pytest.raises(InvalidSignature):
            verify_x509_chain([result.trust_path[0], _der(other_ca[1])])


# --- mutation testing ---------------------------------------------------------


def test_nonce_binding_check_is_load_bearing(
    tmp_path, safetynet_leaf, ca, auth_data
):
    """Mutation test: prove the nonce counterexample catches a real omission.

    A temporary copy of fido2/attestation/android.py is created with the
    nonce binding check removed. The re-signed nonce-mismatch sample,
    which the real implementation rejects, must then be *accepted* -
    demonstrating that test_nonce_mismatch_resigned is sensitive to this
    exact check and would catch its accidental removal.
    """
    import fido2.attestation.android as android_module

    source = Path(android_module.__file__).read_text(encoding="utf-8")
    check = (
        '        if not bytes_eq(expected_nonce, websafe_decode(data["nonce"])):\n'
        '            raise InvalidData("Nonce does not match!")\n'
    )
    assert check in source, "nonce check not found; update the mutation"

    mutated_path = tmp_path / "android_mutated.py"
    mutated_path.write_text(source.replace(check, ""), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "fido2.attestation.android_mutated", mutated_path
    )
    assert spec is not None and spec.loader is not None
    mutated_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mutated_module)

    statement = _safetynet_statement(
        safetynet_leaf, ca, auth_data, nonce=sha256(b"another request")
    )

    # The real implementation rejects the counterexample...
    with pytest.raises(InvalidData, match="Nonce does not match"):
        AndroidSafetynetAttestation().verify(statement, auth_data, CLIENT_DATA_HASH)

    # ...while the mutated copy (nonce check removed) accepts it.
    result = mutated_module.AndroidSafetynetAttestation().verify(
        statement, auth_data, CLIENT_DATA_HASH
    )
    assert result.attestation_type == AttestationType.BASIC
