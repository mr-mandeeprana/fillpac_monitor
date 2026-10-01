#!/usr/bin/env python3
"""
Quick standalone check that the app can reach SQL Server and that
schema.sql has been applied. Run this after setting up .env and before
starting main.py.

Usage: python scripts/check_db_connection.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import load_config  # noqa: E402

try:
    import pyodbc
except ImportError:
    print("ERROR: pyodbc is not installed. Run: pip install pyodbc")
    sys.exit(1)


def main():
    cfg = load_config()
    s = cfg.db_settings()

    if not s.get("enabled"):
        print("database.enabled is false in config.yaml - nothing to check.")
        return

    parts = [f"DRIVER={s['driver']}", f"SERVER={s['server']}", f"DATABASE={s['database']}"]
    if s.get("trusted_connection"):
        parts.append("Trusted_Connection=yes")
    else:
        parts.append(f"UID={s['user']}")
        parts.append(f"PWD={s['password']}")
    parts.append(f"Encrypt={'yes' if s.get('encrypt', True) else 'no'}")
    parts.append(f"TrustServerCertificate={'yes' if s.get('trust_server_certificate', True) else 'no'}")
    conn_str = ";".join(parts)

    print(f"Connecting to server={s['server']} database={s['database']} ...")
    try:
        conn = pyodbc.connect(conn_str, timeout=s.get("connect_timeout_sec", 5))
    except Exception as exc:
        print(f"FAILED to connect: {exc}")
        print("\nCheck: DB_SERVER / DB_NAME / DB_USER / DB_PASSWORD in .env, "
              "that SQL Server allows the auth mode you're using, and that "
              "the ODBC driver named in config.yaml (database.driver) is installed.")
        sys.exit(1)

    print("Connected OK.")
    cur = conn.cursor()
    required_tables = ["Stations", "StatusEvents", "SystemLogs", "CameraHealth"]
    for t in required_tables:
        cur.execute(
            "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?", t
        )
        exists = cur.fetchone()[0] > 0
        print(f"  table dbo.{t}: {'OK' if exists else 'MISSING - run src/db/schema.sql in SSMS'}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
