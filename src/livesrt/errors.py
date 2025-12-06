"""
This module defines the exception hierarchy for the application
"""


class LiveSrtError(Exception):
    """Base exception for all application-specific errors."""

    pass


class TranscribeError(LiveSrtError):
    """Exception raised for errors during the transcription process."""

    pass
