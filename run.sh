#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
REQUIREMENTS="$PROJECT_ROOT/requirements.txt"
REQUIREMENTS_HASH_FILE="$VENV_DIR/.requirements.sha256"
LOG_DIR="$PROJECT_ROOT/.logs"
LOG_FILE="$LOG_DIR/launch.log"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python3 -m venv "$VENV_DIR"
fi

requirements_hash="$(sha256sum "$REQUIREMENTS" | awk '{print $1}')"
installed_hash=""
if [[ -f "$REQUIREMENTS_HASH_FILE" ]]; then
    installed_hash="$(<"$REQUIREMENTS_HASH_FILE")"
fi

if [[ "$requirements_hash" != "$installed_hash" ]]; then
    "$VENV_DIR/bin/python" -m pip install --requirement "$REQUIREMENTS"
    printf '%s\n' "$requirements_hash" > "$REQUIREMENTS_HASH_FILE"
fi

# Activate for the application and any subprocesses it starts.
source "$VENV_DIR/bin/activate"
mkdir -p "$LOG_DIR"
printf '\n[%s] Starting Oswald Globe\n' "$(date --iso-8601=seconds)" >> "$LOG_FILE"

export QT_OPENGL="${QT_OPENGL:-desktop}"
if [[ "${OSTYPE:-}" == linux* ]]; then
    export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-xcb_glx}"
    if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]]; then
        for vendor_json in \
            /usr/share/glvnd/egl_vendor.d/*nvidia*.json \
            /usr/local/share/glvnd/egl_vendor.d/*nvidia*.json \
            /etc/glvnd/egl_vendor.d/*nvidia*.json
        do
            if [[ -f "$vendor_json" ]]; then
                export __EGL_VENDOR_LIBRARY_FILENAMES="$vendor_json"
                break
            fi
        done
    fi
fi

# Keep a complete copy of startup output when terminal scrollback is limited.
set +e
PYTHONUNBUFFERED=1 python "$PROJECT_ROOT/app.py" "$@" 2>&1 | tee -a "$LOG_FILE"
status="${PIPESTATUS[0]}"
set -e

if (( status != 0 )); then
    printf '[%s] Application exited with status %s\n' \
        "$(date --iso-8601=seconds)" "$status" | tee -a "$LOG_FILE" >&2
    printf 'Full launch output is available at %s\n' "$LOG_FILE" >&2
fi

exit "$status"
