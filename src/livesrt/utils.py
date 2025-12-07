"""Utilities for the livesrt package."""

import contextlib
import os
import sys


@contextlib.contextmanager
def ignore_stderr():
    """
    Diverts stderr because by default lots of junk gets logged by alsa and
    friends for no fucking reason.
    """

    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
