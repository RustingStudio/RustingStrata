#!/bin/sh
# Installs the newest RustingStrata release for Linux x86_64 into ~/.local/bin (or $RUSTINGSTRATA_DIR):
#   curl -fsSL https://raw.githubusercontent.com/RustingStudio/RustingStrata/main/install.sh | sh
# Later updates: rustingstrata update
set -eu

[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || {
    echo "error: release binaries are for Linux and Windows x86_64; build from source on $(uname -s) $(uname -m):" >&2
    echo "  https://github.com/RustingStudio/RustingStrata#install" >&2
    exit 1
}
dir=${RUSTINGSTRATA_DIR:-$HOME/.local/bin}
url=https://github.com/RustingStudio/RustingStrata/releases/latest/download/rustingstrata-linux-x86_64
mkdir -p "$dir"
echo "downloading $url"
if command -v curl >/dev/null; then
    curl -fL --progress-bar -o "$dir/rustingstrata.new" "$url"
else
    wget -q --show-progress -O "$dir/rustingstrata.new" "$url"
fi
chmod 755 "$dir/rustingstrata.new"
mv -f "$dir/rustingstrata.new" "$dir/rustingstrata"
echo "installed $("$dir/rustingstrata" --version) to $dir/rustingstrata"
case ":$PATH:" in
    *":$dir:"*) ;;
    *) echo "note: $dir is not in PATH; add it (for bash: echo 'export PATH=\"$dir:\$PATH\"' >> ~/.bashrc) or run $dir/rustingstrata" ;;
esac
echo
echo "Start:  rustingstrata serve --gpu --open   (chat page in the browser; it downloads models)"
echo "   or:  rustingstrata interface            (a few questions in the terminal)"
echo "Update: rustingstrata update"
