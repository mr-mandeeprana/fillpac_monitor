#!/usr/bin/env python3
"""
FillPac Operator Working-Zone Monitor - Entry Point

Usage:
    python main.py
    python main.py --config config/config.yaml --env .env
"""

import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="FillPac Operator Working-Zone Monitor")
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: config/config.yaml)")
    parser.add_argument("--env", default=None, help="Path to .env file (default: .env)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Local import so --help works even if dependencies aren't installed yet.
    from src.app import FillPacApp

    app = FillPacApp(config_path=args.config, env_path=args.env)
    try:
        app.setup()
    except Exception:
        app.logger.exception("Fatal error during setup")
        sys.exit(1)

    app.run()


if __name__ == "__main__":
    main()
