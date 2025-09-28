# --- Variables ---
# Define the lock file name so we can reuse it
REQUIREMENTS_FILE := "requirements.txt"

# Default command to run if no other is specified
default: lint format

# Lock the project dependencies into a requirements.txt file
lock:
    @echo "Locking dependencies from pyproject.toml -> {{REQUIREMENTS_FILE}}..."
    @uv pip compile pyproject.toml -o {{REQUIREMENTS_FILE}}
    @echo "✅ Dependencies locked."

# Sync the virtual environment with the lock file
sync:
    @echo "Syncing environment with {{REQUIREMENTS_FILE}}..."
    @uv pip sync {{REQUIREMENTS_FILE}}
    @echo "✅ Environment synced."

# --- Quality & Maintenance ---
# Lint the code using ruffa
lint:
    @just sync
    @ruff check .

# Format the code using ruff
format:
    @just sync
    @ruff format .

test:
    PYTHONPATH=src pytest tests

# --- Demo ---
# Run the CLI demo that seeds policies and walks through the hybrid memory flow
# Usage: just demo
#        (or) just run-demo
demo:
    PYTHONPATH=src python -m examples.demo

# Friendly alias
run-demo: demo
