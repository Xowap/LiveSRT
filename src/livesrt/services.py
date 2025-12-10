"""Services for the LiveSRT application, e.g., API key storage."""

# Original content of services.py follows
# (Assuming original content is not lost and will be appended or preserved)
from dataclasses import dataclass

import keyring


@dataclass
class ApiKeyStore:
    """
    Utility class that serves to store API keys
    """

    namespace: str
    system: str = "livesrt"

    def key(self, provider: str) -> str:
        """Generates the key name used for storage of this provider"""

        return f"{self.namespace}:{provider}"

    def get(self, provider: str) -> str | None:
        """Gets the API key for a provider, or None if it doesn't exist."""

        return keyring.get_password(self.system, self.key(provider))

    def set(self, provider: str, value: str) -> None:
        """Sets the API key for a provider"""

        keyring.set_password(self.system, self.key(provider), value)
