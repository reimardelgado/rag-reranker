"""Narrow, reversible Open WebUI chat-base redirect for capture task 5.1."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from urllib.parse import urlsplit


CONFIG_KEY = "openai.api_base_urls"


def _read_urls(connection: sqlite3.Connection) -> list[str]:
    row = connection.execute("SELECT value FROM config WHERE key = ?", (CONFIG_KEY,)).fetchone()
    if row is None:
        raise RuntimeError("chat base URL config is missing")
    value = json.loads(row[0]) if isinstance(row[0], str) else row[0]
    if not isinstance(value, list) or not value or not all(isinstance(url, str) for url in value):
        raise RuntimeError("chat base URL config has an unexpected shape")
    for url in value:
        parsed = urlsplit(url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise RuntimeError("refusing to back up a URL containing credentials or opaque data")
    return value


def backup_and_redirect(
    database: Path, backup: Path, expected_url: str, capture_url: str
) -> None:
    if backup.exists():
        raise RuntimeError("backup already exists; rollback or remove it explicitly")
    connection = sqlite3.connect(database)
    try:
        urls = _read_urls(connection)
        if urls != [expected_url]:
            raise RuntimeError("refusing broad mutation: expected exactly one known chat connection")

        backup_connection = sqlite3.connect(backup)
        try:
            backup_connection.execute("CREATE TABLE backup (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            backup_connection.execute(
                "INSERT INTO backup(key, value) VALUES (?, ?)",
                (CONFIG_KEY, json.dumps(urls, separators=(",", ":"))),
            )
            backup_connection.commit()
        finally:
            backup_connection.close()
        os.chmod(backup, 0o600)

        with connection:
            cursor = connection.execute(
                "UPDATE config SET value = ? WHERE key = ?",
                (json.dumps([capture_url], separators=(",", ":")), CONFIG_KEY),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("chat connection update did not affect exactly one row")
    finally:
        connection.close()


def rollback(database: Path, backup: Path, expected_capture_url: str) -> None:
    if not backup.exists():
        raise RuntimeError("rollback backup is missing")
    backup_connection = sqlite3.connect(backup)
    try:
        row = backup_connection.execute(
            "SELECT value FROM backup WHERE key = ?", (CONFIG_KEY,)
        ).fetchone()
        if row is None:
            raise RuntimeError("rollback backup is invalid")
        original_urls = json.loads(row[0])
    finally:
        backup_connection.close()

    connection = sqlite3.connect(database)
    try:
        if _read_urls(connection) != [expected_capture_url]:
            raise RuntimeError("refusing rollback because chat config changed after redirect")
        with connection:
            cursor = connection.execute(
                "UPDATE config SET value = ? WHERE key = ?", (row[0], CONFIG_KEY)
            )
            if cursor.rowcount != 1:
                raise RuntimeError("rollback did not affect exactly one row")
    finally:
        connection.close()
    backup.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("redirect", "rollback"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--database", type=Path, required=True)
        subparser.add_argument("--backup", type=Path, required=True)
    redirect_parser = subparsers.choices["redirect"]
    redirect_parser.add_argument("--expected-url", required=True)
    redirect_parser.add_argument("--capture-url", required=True)
    rollback_parser = subparsers.choices["rollback"]
    rollback_parser.add_argument("--expected-capture-url", required=True)
    args = parser.parse_args()

    if args.command == "redirect":
        backup_and_redirect(args.database, args.backup, args.expected_url, args.capture_url)
    else:
        rollback(args.database, args.backup, args.expected_capture_url)


if __name__ == "__main__":
    main()
