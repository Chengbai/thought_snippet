import tempfile
import shutil
from contextlib import contextmanager


@contextmanager
def temp_directory():
    # Setup: Create temp directory
    temp_dir = tempfile.mkdtemp()
    print(f"Created temp dir: {temp_dir}")

    try:
        yield temp_dir  # Give directory path to user
    finally:
        # Cleanup: Remove temp directory
        shutil.rmtree(temp_dir)
        print(f"Removed temp dir: {temp_dir}")


# Usage
with temp_directory() as tmpdir:
    # tmpdir exists here
    with open(f"{tmpdir}/test.txt", "w") as f:
        f.write("data")
# tmpdir automatically deleted here
