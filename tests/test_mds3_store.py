import json
from base64 import b64encode
from datetime import date, datetime, timedelta

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from fido2.mds3 import (
    BlobConflict,
    CorruptState,
    InvalidBlob,
    LocalMdsStore,
    MdsStatus,
    MetadataNotCurrent,
    StaleBlob,
    UpdateOutcome,
)
from fido2.utils import websafe_decode, websafe_encode

AAGUID_A = bytes.fromhex("0132d110bf4e4208a403ab4f5f12efe5")
AAGUID_B = bytes.fromhex("11111111111111111111111111111111")
AAGUID_MISSING = bytes.fromhex("22222222222222222222222222222222")


class MutableClock:
    def __init__(self, today):
        self.today = today

    def __call__(self):
        return self.today


def _root_ca():
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test MDS Root")])
    now = datetime(2026, 1, 1)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=3650))
        .sign(key, hashes.SHA256())
    )
    return key, cert.public_bytes(serialization.Encoding.DER)


@pytest.fixture
def signer():
    key, root = _root_ca()

    def make_blob(no, next_update, aaguids=(AAGUID_A,), revoked=(), material=None):
        signing_key, signing_cert = material or (key, root)
        cert_der = b64encode(signing_cert).decode("ascii")
        entries = [
            {
                "aaguid": aaguid.hex(),
                "statusReports": [
                    {
                        "status": (
                            "REVOKED" if aaguid in revoked else "FIDO_CERTIFIED"
                        )
                    }
                ],
                "timeOfLastStatusChange": "2026-01-01",
            }
            for aaguid in aaguids
        ]
        payload = {
            "legalHeader": "test header",
            "no": no,
            "nextUpdate": next_update.isoformat(),
            "entries": entries,
        }
        header = {"alg": "ES256", "typ": "JWT", "x5c": [cert_der]}
        message = (
            websafe_encode(json.dumps(header, separators=(",", ":")).encode())
            + "."
            + websafe_encode(json.dumps(payload, separators=(",", ":")).encode())
        )
        signature = signing_key.sign(
            message.encode("ascii"), ec.ECDSA(hashes.SHA256())
        )
        return (message + "." + websafe_encode(signature)).encode("ascii")

    make_blob.root = root
    return make_blob


@pytest.fixture
def clock():
    return MutableClock(date(2026, 6, 1))


def make_store(tmp_path, signer, clock):
    return LocalMdsStore(tmp_path / "mds" / "state.json", signer.root, clock=clock)


def test_empty_store(tmp_path, signer, clock):
    store = make_store(tmp_path, signer, clock)
    assert store.status is MdsStatus.EMPTY
    assert store.is_current() is False
    assert store.sequence_number is None
    assert store.next_update is None
    assert store.payload is None
    assert store.find_entry_by_aaguid(AAGUID_A) is None
    with pytest.raises(MetadataNotCurrent):
        store.verifier


def test_import_and_status_current(tmp_path, signer, clock):
    store = make_store(tmp_path, signer, clock)
    blob = signer(1, date(2026, 12, 31))

    assert store.update(blob) is UpdateOutcome.APPLIED

    assert store.status is MdsStatus.CURRENT
    assert store.is_current() is True
    assert store.sequence_number == 1
    assert store.next_update == date(2026, 12, 31)
    assert store.payload.no == 1
    assert store.find_entry_by_aaguid(AAGUID_A).aaguid == AAGUID_A
    assert store.find_entry_by_aaguid(AAGUID_MISSING) is None
    assert store.verifier.find_entry_by_aaguid(AAGUID_A) is not None


def test_same_no_same_content_is_idempotent(tmp_path, signer, clock):
    store = make_store(tmp_path, signer, clock)
    blob = signer(1, date(2026, 12, 31))
    assert store.update(blob) is UpdateOutcome.APPLIED
    assert store.update(blob) is UpdateOutcome.UNCHANGED
    assert store.sequence_number == 1


def test_same_no_different_content_rejected(tmp_path, signer, clock):
    store = make_store(tmp_path, signer, clock)
    store.update(signer(1, date(2026, 12, 31), aaguids=(AAGUID_A,)))
    conflicting = signer(1, date(2026, 12, 31), aaguids=(AAGUID_B,))

    with pytest.raises(BlobConflict) as exc:
        store.update(conflicting)
    assert exc.value.no == 1

    assert store.sequence_number == 1
    assert store.find_entry_by_aaguid(AAGUID_A) is not None
    assert store.find_entry_by_aaguid(AAGUID_B) is None


def test_lower_no_cannot_overwrite(tmp_path, signer, clock):
    store = make_store(tmp_path, signer, clock)
    store.update(signer(3, date(2026, 12, 31)))
    older = signer(2, date(2026, 12, 31))

    with pytest.raises(StaleBlob) as exc:
        store.update(older)
    assert exc.value.incoming_no == 2
    assert exc.value.current_no == 3
    assert store.sequence_number == 3


def test_higher_no_advances(tmp_path, signer, clock):
    store = make_store(tmp_path, signer, clock)
    store.update(signer(1, date(2026, 6, 2), aaguids=(AAGUID_A,)))
    outcome = store.update(
        signer(2, date(2026, 12, 31), aaguids=(AAGUID_B,))
    )
    assert outcome is UpdateOutcome.APPLIED
    assert store.sequence_number == 2
    assert store.find_entry_by_aaguid(AAGUID_B) is not None


def test_untrusted_blob_rejected_and_state_kept(tmp_path, signer, clock):
    store = make_store(tmp_path, signer, clock)
    store.update(signer(1, date(2026, 12, 31)))

    # A blob signed by a different CA must not validate against our trust root.
    other_key, other_root = _root_ca()
    forged = signer(
        2, date(2026, 12, 31), material=(other_key, other_root)
    )
    other_store = LocalMdsStore(tmp_path / "other.json", other_root, clock=clock)
    with pytest.raises(InvalidBlob):
        other_store.update(signer(1, date(2026, 12, 31)))
    assert other_store.status is MdsStatus.EMPTY

    with pytest.raises(InvalidBlob):
        store.update(forged)
    assert store.sequence_number == 1
    assert store.status is MdsStatus.CURRENT


def test_malformed_blob_rejected(tmp_path, signer, clock):
    store = make_store(tmp_path, signer, clock)
    with pytest.raises(InvalidBlob):
        store.update(b"not a jwt at all")
    assert store.status is MdsStatus.EMPTY


def test_tampered_signature_rejected(tmp_path, signer, clock):
    store = make_store(tmp_path, signer, clock)
    blob = bytearray(signer(1, date(2026, 12, 31)))
    signature = blob.rpartition(b".")[2]
    blob[len(blob) - len(signature) + 4] ^= 0x01
    with pytest.raises(InvalidBlob):
        store.update(bytes(blob))
    assert store.status is MdsStatus.EMPTY


def test_expired_blob_queryable_but_not_for_admission(tmp_path, signer, clock):
    store = make_store(tmp_path, signer, clock)
    store.update(signer(1, date(2026, 5, 31)))

    assert store.status is MdsStatus.EXPIRED
    assert store.is_current() is False
    assert store.sequence_number == 1
    # Existing devices remain queryable ...
    assert store.find_entry_by_aaguid(AAGUID_A) is not None
    # ... but admission of new devices is refused at the interface.
    with pytest.raises(MetadataNotCurrent) as exc:
        store.verifier
    assert exc.value.args[0] == MdsStatus.EXPIRED.value

    clock.today = date(2026, 5, 31)
    assert store.status is MdsStatus.CURRENT
    assert store.verifier.find_entry_by_aaguid(AAGUID_A) is not None

    clock.today = date(2026, 6, 1)
    assert store.status is MdsStatus.EXPIRED


def test_next_update_boundary_day_is_current(tmp_path, signer, clock):
    store = make_store(tmp_path, signer, clock)
    store.update(signer(1, date(2026, 6, 1)))

    clock.today = date(2026, 6, 1)
    assert store.status is MdsStatus.CURRENT
    assert store.verifier is not None

    clock.today = date(2026, 6, 2)
    assert store.status is MdsStatus.EXPIRED


def test_expired_store_accepts_newer_blob_and_recovers(tmp_path, signer, clock):
    store = make_store(tmp_path, signer, clock)
    store.update(signer(1, date(2026, 5, 31)))
    assert store.status is MdsStatus.EXPIRED

    assert store.update(signer(2, date(2026, 12, 31))) is UpdateOutcome.APPLIED
    assert store.status is MdsStatus.CURRENT
    assert store.sequence_number == 2
    assert store.verifier is not None


def test_persistence_survives_restart(tmp_path, signer, clock):
    path = tmp_path / "state.json"
    store = LocalMdsStore(path, signer.root, clock=clock)
    blob = signer(7, date(2026, 12, 31), aaguids=(AAGUID_A, AAGUID_B))
    store.update(blob)

    reopened = LocalMdsStore(path, signer.root, clock=clock)
    assert reopened.sequence_number == 7
    assert reopened.status is MdsStatus.CURRENT
    assert reopened.find_entry_by_aaguid(AAGUID_B) is not None
    assert reopened.verifier.find_entry_by_aaguid(AAGUID_A) is not None
    # The exact same authenticated bytes can be re-imported after a restart.
    assert reopened.update(blob) is UpdateOutcome.UNCHANGED


def test_persisted_expired_blob_reopens_expired(tmp_path, signer):
    path = tmp_path / "state.json"
    fixed_clock = MutableClock(date(2026, 1, 1))
    store = LocalMdsStore(path, signer.root, clock=fixed_clock)
    store.update(signer(1, date(2026, 2, 1)))

    later_clock = MutableClock(date(2026, 3, 1))
    reopened = LocalMdsStore(path, signer.root, clock=later_clock)
    assert reopened.status is MdsStatus.EXPIRED
    with pytest.raises(MetadataNotCurrent):
        reopened.verifier
    assert reopened.find_entry_by_aaguid(AAGUID_A) is not None


def test_corrupted_state_detected(tmp_path, signer, clock):
    path = tmp_path / "state.json"
    store = LocalMdsStore(path, signer.root, clock=clock)
    store.update(signer(1, date(2026, 12, 31)))

    raw = json.loads(path.read_text())
    raw["no"] = 999
    path.write_text(json.dumps(raw))
    with pytest.raises(CorruptState):
        LocalMdsStore(path, signer.root, clock=clock)


def test_blob_tampering_in_state_detected(tmp_path, signer, clock):
    path = tmp_path / "state.json"
    LocalMdsStore(path, signer.root, clock=clock).update(
        signer(1, date(2026, 12, 31))
    )
    raw = json.loads(path.read_text())
    decoded = bytearray(websafe_decode(raw["blob"]))
    decoded[-2] ^= 0x01
    raw["blob"] = websafe_encode(bytes(decoded))
    path.write_text(json.dumps(raw))
    with pytest.raises(CorruptState):
        LocalMdsStore(path, signer.root, clock=clock)


def test_truncated_state_detected(tmp_path, signer, clock):
    path = tmp_path / "state.json"
    LocalMdsStore(path, signer.root, clock=clock).update(
        signer(1, date(2026, 12, 31))
    )
    path.write_text("{not valid json")
    with pytest.raises(CorruptState):
        LocalMdsStore(path, signer.root, clock=clock)


def test_trust_root_is_required(tmp_path, signer, clock):
    with pytest.raises(TypeError):
        LocalMdsStore(tmp_path / "state.json", None, clock=clock)


def test_entry_filter_applies_to_verifier_only(tmp_path, signer, clock):
    store = LocalMdsStore(
        tmp_path / "state.json",
        signer.root,
        clock=clock,
        entry_filter=lambda e: e.aaguid != AAGUID_A,
    )
    store.update(signer(1, date(2026, 12, 31), aaguids=(AAGUID_A, AAGUID_B)))
    assert store.verifier.find_entry_by_aaguid(AAGUID_A) is None
    assert store.verifier.find_entry_by_aaguid(AAGUID_B) is not None


def test_revoked_entries_filtered_by_default(tmp_path, signer, clock):
    store = make_store(tmp_path, signer, clock)
    store.update(
        signer(1, date(2026, 12, 31), aaguids=(AAGUID_A,), revoked=(AAGUID_A,))
    )
    assert store.verifier.find_entry_by_aaguid(AAGUID_A) is None


def test_failed_update_leaves_index_usable(tmp_path, signer, clock):
    store = make_store(tmp_path, signer, clock)
    store.update(signer(1, date(2026, 12, 31), aaguids=(AAGUID_A,)))

    with pytest.raises(StaleBlob):
        store.update(signer(0, date(2026, 12, 31), aaguids=(AAGUID_B,)))
    with pytest.raises(BlobConflict):
        store.update(signer(1, date(2026, 12, 31), aaguids=(AAGUID_B,)))

    verifier = store.verifier
    assert verifier.find_entry_by_aaguid(AAGUID_A) is not None
    assert store.find_entry_by_aaguid(AAGUID_B) is None
    assert store.sequence_number == 1


def test_find_entry_by_chain(tmp_path, signer, clock):
    store = make_store(tmp_path, signer, clock)
    store.update(signer(1, date(2026, 12, 31)))
    # No SKI entries in the test blob, so a chain lookup simply misses.
    assert store.find_entry_by_chain([signer.root]) is None
