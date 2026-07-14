"""Integration test: load sessions from offset pipeline input, sample random
UUIDs, fetch full session JSON via HTTP API, and validate the conversion.

Usage:
    # Dry-run: validate models from offset pipeline input only (no network)
    python scripts/integration_test.py --dry-run

    # Full integration: sample UUIDs, fetch via API, convert, validate
    python scripts/integration_test.py --sample-size 10

    # Custom API endpoint
    python scripts/integration_test.py --api-url "http://host:port/api/..." --sample-size 5
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Point to the source package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chrys_trace_analysis.config import load_config
from chrys_trace_analysis.loader import load_user_files
from chrys_trace_analysis.models import Session, UserData
from chrys_trace_analysis.mongo_loader import mongo_docs_to_user_data

# Default API endpoint — matches session_request.py
DEFAULT_API_URL = "http://lingxi-stats.rnd.huawei.com:8042/api/chrys/query/session"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("integration_test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_session_uuids(users: list[UserData]) -> dict[str, str]:
    """Build session_uuid → user_name mapping from UserData list."""
    mapping: dict[str, str] = {}
    for user in users:
        for session in user.sessions:
            mapping[session.session_uuid] = user.user_name
    return mapping


def _pick_random_uuids(
    uuid_map: dict[str, str],
    sample_size: int,
    seed: int = 42,
) -> list[str]:
    """Randomly pick session UUIDs from the map."""
    all_uuids = list(uuid_map.keys())
    if len(all_uuids) <= sample_size:
        return all_uuids
    rng = random.Random(seed)
    return rng.sample(all_uuids, sample_size)


def _validate_conversion(
    label: str,
    result_users: list[UserData],
) -> dict:
    """Validate the converted UserData and return a report dict."""
    report: dict = {
        "label": label,
        "user_count": len(result_users),
        "session_count": sum(len(u.sessions) for u in result_users),
        "errors": [],
        "checks_passed": 0,
        "checks_failed": 0,
    }

    for user in result_users:
        if not isinstance(user, UserData):
            report["errors"].append(f"Not a UserData: {type(user)}")
            report["checks_failed"] += 1
            continue

        report["checks_passed"] += 1

        for session in user.sessions:
            assert isinstance(session, Session), f"Not a Session: {type(session)}"
            assert isinstance(session.session_uuid, str), "session_uuid not string"
            assert isinstance(session.session_abstract, list), "session_abstract not list"
            assert isinstance(session.turns, list), "turns not list"
            assert isinstance(session.meta, dict), "meta not dict"

            if session.turns and session.session_abstract:
                if len(session.turns) != len(session.session_abstract):
                    report["errors"].append(
                        f"session {session.session_uuid}: "
                        f"turns={len(session.turns)} != abstract={len(session.session_abstract)}"
                    )
                    report["checks_failed"] += 1

            for turn in session.turns:
                for msg in turn.messages:
                    assert msg.role in ("user", "assistant", "system", "tool"), \
                        f"Bad role: {msg.role}"

    return report


def _print_report(report: dict) -> None:
    """Pretty-print a validation report."""
    print(f"\n{'='*60}")
    print(f"  {report['label']}")
    print(f"{'='*60}")
    print(f"  Users:    {report['user_count']}")
    print(f"  Sessions: {report['session_count']}")
    print(f"  Checks passed: {report['checks_passed']}")
    if report["checks_failed"]:
        print(f"  Checks FAILED: {report['checks_failed']}")
    if report["errors"]:
        for err in report["errors"][:10]:
            print(f"    ⚠ {err}")
        if len(report["errors"]) > 10:
            print(f"    ... and {len(report['errors']) - 10} more errors")
    print()


# ---------------------------------------------------------------------------
# Dry-run: local model validation only
# ---------------------------------------------------------------------------


def run_dry_run(data_dir: Path) -> bool:
    """Load from offset pipeline input and validate the data model.

    This exercises the model layer and mongo_docs_to_user_data using
    existing JSON data — no network access needed.
    """
    print("\n" + "=" * 60)
    print("  DRY-RUN: Validate session models from offset pipeline input")
    print("=" * 60)

    users = load_user_files(data_dir)
    if not users:
        logger.error("No users loaded from %s", data_dir)
        return False

    uuid_map = _collect_session_uuids(users)
    logger.info("Loaded %d users, %d sessions", len(users), len(uuid_map))

    # Validate the loaded data
    report = _validate_conversion("Offset input validation", users)
    _print_report(report)

    # Re-parse sessions with turn data through mongo_docs_to_user_data
    sessions_with_turns = sum(
        1 for u in users for s in u.sessions if s.turns
    )
    if sessions_with_turns > 0:
        sessions_by_user: dict[str, list[dict]] = {}
        for user in users:
            docs: list[dict] = []
            for s in user.sessions:
                if not s.turns:
                    continue
                messages: list[dict] = []
                for turn in s.turns:
                    for msg in turn.messages:
                        messages.append({
                            "role": msg.role,
                            "contents": [c.model_dump() for c in msg.contents],
                            "additional_properties": msg.additional_properties,
                            "message_id": msg.message_id,
                        })
                    messages.append({
                        "role": "assistant",
                        "contents": [],
                        "additional_properties": {"_chrys_kind": "turn"},
                    })
                docs.append({
                    "uuid": s.session_uuid,
                    "messages": messages,
                    "mcp_tools": [{"tool_name": t} for t in s.mcp_tools],
                    "skills": [{"skill_name": sk} for sk in s.skills],
                    "meta": s.meta,
                    "mr_relation": {},
                })
            sessions_by_user[user.user_name] = docs

        reconverted = mongo_docs_to_user_data(sessions_by_user)
        reconvert_report = _validate_conversion(
            "Re-converted via mongo_docs_to_user_data", reconverted,
        )
        _print_report(reconvert_report)

        if sessions_with_turns != reconvert_report["session_count"]:
            logger.error("Session count mismatch: %d (with turns) vs %d (reconverted)",
                         sessions_with_turns, reconvert_report["session_count"])
            return False

        for user in reconverted:
            UserData.model_validate_json(user.model_dump_json())
    else:
        logger.info("No sessions with turn data — skipping reconvert step.")
        for user in users:
            UserData.model_validate_json(user.model_dump_json())

    print("  ✓ Dry-run PASSED — model validation and round-trip OK\n")
    return True


# ---------------------------------------------------------------------------
# HTTP fetch: sample UUIDs → call API → convert → validate
# ---------------------------------------------------------------------------


def _fetch_one_session(api_url: str, uuid: str, timeout: int = 30) -> dict | None:
    """Fetch a single session JSON from the HTTP API.

    Returns the parsed JSON dict, or None on failure.
    """
    payload = json.dumps({
        "filter_key": "uuid",
        "filter_value": uuid,
    }).encode("utf-8")

    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        logger.warning("HTTP %d for UUID %s: %s", exc.code, uuid, exc.reason)
    except urllib.error.URLError as exc:
        logger.warning("Connection error for UUID %s: %s", uuid, exc.reason)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to fetch/parse UUID %s: %s", uuid, exc)

    return None


def _extract_user_id(doc: dict) -> str:
    """Extract user_id from a session document."""
    meta = doc.get("meta", {})
    if isinstance(meta, dict):
        return meta.get("user_id", "unknown")
    return "unknown"


def run_http_test(
    uuid_map: dict[str, str],
    api_url: str,
    sample_size: int,
) -> bool:
    """Sample UUIDs, fetch full session JSON via HTTP API, convert and validate."""
    print("\n" + "=" * 60)
    print("  HTTP INTEGRATION: Fetch sessions from API")
    print("=" * 60)

    # Step 1: Pick random UUIDs
    sample_uuids = _pick_random_uuids(uuid_map, sample_size)
    logger.info("Sampled %d session UUIDs", len(sample_uuids))

    # Step 2: Fetch each UUID from the API
    sessions_by_user: dict[str, list[dict]] = {}
    fetched = 0
    failed = 0

    for i, uuid in enumerate(sample_uuids, 1):
        logger.info("[%d/%d] Fetching %s ...", i, len(sample_uuids), uuid)
        doc = _fetch_one_session(api_url, uuid)
        if doc is None:
            failed += 1
            continue
        fetched += 1

        user_id = _extract_user_id(doc)
        sessions_by_user.setdefault(user_id, []).append(doc)

    logger.info("Fetched: %d/%d (failed: %d)", fetched, len(sample_uuids), failed)
    logger.info("Unique users with data: %d", len(sessions_by_user))

    if not sessions_by_user:
        logger.warning("No sessions fetched — nothing to validate.")
        return True

    # Step 3: Convert through the full pipeline
    users = mongo_docs_to_user_data(sessions_by_user)
    logger.info("Converted to %d UserData objects", len(users))

    # Step 4: Validate
    report = _validate_conversion("HTTP API fetch result", users)
    _print_report(report)

    # Step 5: Check UUID coverage
    sampled_set = set(sample_uuids)
    fetched_uuids: set[str] = set()
    for user in users:
        for session in user.sessions:
            fetched_uuids.add(session.session_uuid)

    matched = sampled_set & fetched_uuids
    missing = sampled_set - fetched_uuids
    logger.info("UUID match: %d/%d found in API responses", len(matched), len(sampled_set))
    if missing:
        logger.warning("Missing UUIDs (%d): %s", len(missing), list(missing)[:5])

    # Step 6: JSON round-trip
    for user in users:
        UserData.model_validate_json(user.model_dump_json())

    if report["checks_failed"] > 0:
        logger.error("HTTP integration FAILED — %d checks failed",
                     report["checks_failed"])
        return False

    print("  ✓ HTTP integration test PASSED\n")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integration test: sample UUIDs → HTTP fetch → model conversion",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate models from offset pipeline input (no network access needed)",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml (for data_dir, etc.)",
    )
    parser.add_argument(
        "--api-url", type=str, default=DEFAULT_API_URL,
        help="Full URL of the session query API endpoint",
    )
    parser.add_argument(
        "--sample-size", type=int, default=10,
        help="Number of session UUIDs to sample and fetch (default: 10)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Data directory (overrides config.paths.data_dir)",
    )

    args = parser.parse_args()

    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif Path(args.config).exists():
        cfg = load_config(args.config)
        data_dir = cfg.paths.data_dir
    else:
        data_dir = Path("./data")

    if not data_dir.exists():
        logger.error("Data directory does not exist: %s", data_dir)
        logger.info("  Create it or use --data-dir to point to existing data.")
        sys.exit(1)

    all_passed = True

    # Always run dry-run validation first
    if not run_dry_run(data_dir):
        all_passed = False

    # Optionally run HTTP integration test
    if not args.dry_run:
        users = load_user_files(data_dir)
        uuid_map = _collect_session_uuids(users)
        if not uuid_map:
            logger.error("No sessions in input data — cannot sample.")
            all_passed = False
        else:
            if not run_http_test(uuid_map, args.api_url, args.sample_size):
                all_passed = False

    if all_passed:
        print("=" * 60)
        print("  ALL TESTS PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print("  SOME TESTS FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
