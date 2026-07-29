"""Exception hierarchy compatible with argon2-cffi."""


class Argon2Error(Exception):
    pass


class VerificationError(Argon2Error):
    pass


class VerifyMismatchError(VerificationError):
    pass


class HashingError(Argon2Error):
    pass


class InvalidHashError(ValueError):
    pass


class UnsupportedParametersError(ValueError):
    pass


InvalidHash = InvalidHashError
