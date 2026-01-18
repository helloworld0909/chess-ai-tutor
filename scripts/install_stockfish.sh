#!/bin/bash
# Install Stockfish 17 with NNUE support

set -e

STOCKFISH_VERSION="stockfish-ubuntu-x86-64-avx2"
STOCKFISH_URL="https://github.com/official-stockfish/Stockfish/releases/download/sf_17/${STOCKFISH_VERSION}.tar"
INSTALL_DIR="${HOME}/.local/bin"
NNUE_URL="https://tests.stockfishchess.org/api/nn/nn-1111cefa1111.nnue"

echo "=== Installing Stockfish 17 ==="

# Create install directory
mkdir -p "$INSTALL_DIR"

# Download and extract Stockfish
echo "Downloading Stockfish 17..."
cd /tmp
wget -q "$STOCKFISH_URL" -O stockfish.tar
tar -xf stockfish.tar

# Find the binary
STOCKFISH_BIN=$(find . -name "stockfish*" -type f -executable 2>/dev/null | head -1)

if [ -z "$STOCKFISH_BIN" ]; then
    echo "Error: Could not find Stockfish binary"
    exit 1
fi

# Install binary
echo "Installing to $INSTALL_DIR..."
cp "$STOCKFISH_BIN" "$INSTALL_DIR/stockfish"
chmod +x "$INSTALL_DIR/stockfish"

# Download NNUE weights (optional - Stockfish 17 has embedded weights)
echo "Downloading NNUE weights..."
mkdir -p "${HOME}/.stockfish"
wget -q "$NNUE_URL" -O "${HOME}/.stockfish/nn-1111cefa1111.nnue" || true

# Cleanup
rm -f stockfish.tar
rm -rf stockfish-*

# Verify installation
echo ""
echo "=== Verifying Installation ==="
if "$INSTALL_DIR/stockfish" <<< "uci" | grep -q "Stockfish"; then
    echo "Stockfish 17 installed successfully!"
    "$INSTALL_DIR/stockfish" <<< "uci" | grep "id name"
else
    echo "Error: Stockfish installation verification failed"
    exit 1
fi

echo ""
echo "Add to PATH: export PATH=\"\$HOME/.local/bin:\$PATH\""
echo "Or set STOCKFISH_PATH=$INSTALL_DIR/stockfish"
