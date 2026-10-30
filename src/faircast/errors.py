"""FairCast error type.

Every input-validation failure raises :class:`FairCastError` with a specific
message. FairCast never silently substitutes, drops, or repairs an
in-domain-but-invalid input: the caller is told exactly what is wrong.
"""


class FairCastError(ValueError):
    """Raised for invalid inputs, unusable series, or unmet preconditions."""
