import subprocess
import sys


def test_livesrt_version():
    try:
        # Expecting a successful exit for --version, and then checking output
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "livesrt", "--version"],
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        )

        # Check that the version string is present in stdout and exit code is 0
        assert "livesrt version:" in result.stdout.strip()
        assert result.returncode == 0

    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        raise RuntimeError(error_message) from e


if __name__ == "__main__":
    test_livesrt_version()
