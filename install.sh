#!/usr/bin/env bash
# Build curds and install it onto $PATH.
#
# Defaults to /usr/local/bin (or whatever PREFIX is set to). Falls back to
# ~/.local/bin if /usr/local/bin isn't writable. Both shell-quotes the
# inputs and echoes every step so you can audit what it did.

set -euo pipefail

cd "$(dirname "$0")"

PREFIX="${PREFIX:-}"
BIN_NAME="curds"

# Sanity-check go is on PATH.
if ! command -v go >/dev/null 2>&1; then
    echo "error: go toolchain not found on PATH" >&2
    echo "       install Go from https://go.dev/dl or via your package manager" >&2
    exit 1
fi

echo "==> building $BIN_NAME ($(go version | awk '{print $3, $4}'))"
go build -o "./$BIN_NAME" ./cmd/curds

# Pick an install dir.
if [[ -z "$PREFIX" ]]; then
    if [[ -w /usr/local/bin ]]; then
        PREFIX="/usr/local/bin"
    elif [[ -d /usr/local/bin ]]; then
        # /usr/local/bin exists but isn't writable — needs sudo.
        PREFIX="/usr/local/bin"
        SUDO="sudo"
    else
        PREFIX="$HOME/.local/bin"
        mkdir -p "$PREFIX"
    fi
fi

DEST="$PREFIX/$BIN_NAME"

echo "==> installing to $DEST"
${SUDO:-} install -m 0755 "./$BIN_NAME" "$DEST"

# Helpful PATH check.
case ":$PATH:" in
    *":$PREFIX:"*) ;;
    *)
        cat <<EOF
note: $PREFIX is not on your \$PATH.
      add this to ~/.config/fish/config.fish:
          set -gx PATH \$PATH $PREFIX
      or to ~/.bashrc / ~/.zshrc:
          export PATH="\$PATH:$PREFIX"
EOF
        ;;
esac

echo "==> done. try: curds -h"
