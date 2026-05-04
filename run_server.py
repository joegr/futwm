#!/usr/bin/env python3
"""Launch the soccer world-model dashboard server."""
from soccer_model.server import run

if __name__ == "__main__":
    run(host="127.0.0.1", port=5050, debug=True)
