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

"""Server-side admission policies for authenticator extension data.

The data returned in a registration response comes from two distinct trust
boundaries:

* ``authenticator_extensions`` is the CBOR map embedded in the
  ``authenticatorData`` byte string. It is covered by the attestation
  signature and therefore attested by the authenticator.
* ``client_extensions`` mirrors the browser's ``clientExtensionResults`` and
  is only self-reported by the client. It is never part of the signed data.

A policy must base security decisions on the attested data only. The client
reported values are provided for context, but a claim present there can never
substitute for a missing or insufficient authenticator extension.

Only CBOR decoding of the authenticator extension map is performed, using the
same constrained decoder as the rest of the library. Unknown, non-required
extensions are preserved as plain decoded values for the caller; no custom
deserialization hooks are invoked.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Protocol, runtime_checkable


def _freeze_mapping(value: Any) -> Any:
    """Return a read-only view of a decoded CBOR mapping.

    Nested mappings are wrapped as well so a policy cannot mutate any data the
    server parsed from the signed authenticator data. Non-mapping values are
    returned unchanged.
    """
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_mapping(item) for key, item in value.items()}
        )
    return value


class ExtensionPolicyError(ValueError):
    """Base class for errors raised when an extension policy rejects a
    credential."""


class MissingExtensionError(ExtensionPolicyError):
    """A required authenticator extension is absent from the attested
    extension data.

    A value present only in the client reported extensions does not count.
    """

    def __init__(self, extension: str):
        super().__init__(
            f'Required authenticator extension "{extension}" is not included '
            "in the signed authenticator data (client reported values are not "
            "attested)."
        )
        self.extension = extension


class InvalidExtensionValueError(ExtensionPolicyError):
    """An attested extension value has an unexpected type."""

    def __init__(self, extension: str, value: Any):
        super().__init__(
            f'Authenticator extension "{extension}" has an invalid value: '
            f"{value!r}."
        )
        self.extension = extension
        self.value = value


class InsufficientExtensionLevelError(ExtensionPolicyError):
    """An attested extension value does not reach the required level."""

    def __init__(self, extension: str, required: int, actual: int):
        super().__init__(
            f'Authenticator extension "{extension}" level {actual} is below '
            f"the required level {required}."
        )
        self.extension = extension
        self.required = required
        self.actual = actual


@dataclass(frozen=True)
class RegistrationExtensionEvidence:
    """Read-only evidence about extensions, as observed for one registration.

    :ivar requested_extensions: The extensions mapping the server sent in the
        ``PublicKeyCredentialCreationOptions`` for this registration, or an
        empty mapping when none were requested.
    :ivar authenticator_extensions: The extension map decoded from the signed
        ``authenticatorData``, or an empty mapping when the extension data flag
        was not set. Nested mappings are read-only.
    :ivar client_extensions: The client self-reported
        ``clientExtensionResults``. These values are not covered by any
        signature and must never be treated as attested.
    """

    requested_extensions: Mapping[str, Any]
    authenticator_extensions: Mapping[str, Any]
    client_extensions: Mapping[str, Any]


@runtime_checkable
class ExtensionPolicy(Protocol):
    """Admission policy evaluated after the standard registration checks.

    Implementations must raise an :class:`ExtensionPolicyError` (or any other
    exception) to reject the credential, and return normally to accept it.
    Policies only observe the current response: the evidence is read-only and
    the raw ``authenticatorData`` used for signature verification is never
    exposed for mutation.
    """

    def check_registration_extensions(
        self, evidence: RegistrationExtensionEvidence
    ) -> None: ...


class AbstractExtensionPolicy(abc.ABC):
    """Convenience base class for implementing extension policies."""

    @abc.abstractmethod
    def check_registration_extensions(
        self, evidence: RegistrationExtensionEvidence
    ) -> None:
        """Validate the extension evidence of a single registration.

        :param evidence: Read-only extension data for this registration.
        :raises ExtensionPolicyError: If the credential must be rejected.
        """


# A plain callable with the same signature as the policy method is accepted
# wherever an ExtensionPolicy can be supplied.
RegistrationExtensionPolicy = (
    ExtensionPolicy | Callable[[RegistrationExtensionEvidence], None]
)


def evaluate_registration_extension_policy(
    policy: RegistrationExtensionPolicy,
    evidence: RegistrationExtensionEvidence,
) -> None:
    """Invoke either an ExtensionPolicy or an equivalent plain callable."""
    if callable(policy) and not hasattr(policy, "check_registration_extensions"):
        plain_callable: Callable[[RegistrationExtensionEvidence], None] = policy
        plain_callable(evidence)
    else:
        policy.check_registration_extensions(evidence)


class CredProtectPolicy(AbstractExtensionPolicy):
    """Require a minimum credProtect (credential protection) level.

    The credProtect CTAP2 extension is attested in ``authenticatorData`` as a
    CBOR map containing ``{"credProtect": <1|2|3>}``. The levels are:

    1. ``userVerificationOptional``
    2. ``userVerificationOptionalWithCredentialIDList``
    3. ``userVerificationRequired``

    Only the attested integer is consulted. A browser reporting a
    credProtect-related value in ``clientExtensionResults`` cannot satisfy the
    requirement on its own.

    :param minimum_level: The minimum acceptable credProtect level (1-3).
    """

    NAME = "credProtect"

    def __init__(self, minimum_level: int):
        if not isinstance(minimum_level, int) or isinstance(minimum_level, bool):
            raise TypeError("minimum_level must be an integer.")
        if not 1 <= minimum_level <= 3:
            raise ValueError("minimum_level must be between 1 and 3.")
        self.minimum_level = minimum_level

    def check_registration_extensions(
        self, evidence: RegistrationExtensionEvidence
    ) -> None:
        if self.NAME not in evidence.authenticator_extensions:
            raise MissingExtensionError(self.NAME)

        level = evidence.authenticator_extensions[self.NAME]
        # bool is a subclass of int in Python and must not be accepted.
        if not isinstance(level, int) or isinstance(level, bool):
            raise InvalidExtensionValueError(self.NAME, level)

        if level < self.minimum_level:
            raise InsufficientExtensionLevelError(self.NAME, self.minimum_level, level)
