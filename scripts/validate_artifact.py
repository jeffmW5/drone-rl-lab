#!/usr/bin/env python3
"""
Validate JSON artifacts against drone-rl-lab schemas.

Usage:
    python scripts/validate_artifact.py inbox/tasks/exp_066.json
    python scripts/validate_artifact.py state/jobs/job_exp_066_001.json
    python scripts/validate_artifact.py --schema task inbox/tasks/*.json
    python scripts/validate_artifact.py --all          # validate all known artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
SCHEMAS_DIR = REPO_DIR / "schemas"

# Map schema names to file paths
SCHEMA_MAP = {
    "task": SCHEMAS_DIR / "task.schema.json",
    "job_state": SCHEMAS_DIR / "job_state.schema.json",
}

# Map artifact directories to their schema
DIR_SCHEMA_MAP = {
    "inbox/tasks": "task",
    "state/jobs": "job_state",
}


def load_schema(schema_name: str) -> dict:
    """Load a schema by name."""
    path = SCHEMA_MAP.get(schema_name)
    if not path or not path.is_file():
        raise FileNotFoundError(f"Schema not found: {schema_name}")
    return json.loads(path.read_text())


def guess_schema_name(artifact_path: Path) -> str | None:
    """Guess which schema applies based on the artifact's directory."""
    rel = artifact_path.resolve().relative_to(REPO_DIR)
    for dir_prefix, schema_name in DIR_SCHEMA_MAP.items():
        if str(rel).startswith(dir_prefix):
            return schema_name
    return None


def validate_required(artifact: dict, schema: dict) -> list[str]:
    """Check required fields are present."""
    errors = []
    for field in schema.get("required", []):
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    return errors


def validate_enum(value, prop_schema: dict, field_name: str) -> list[str]:
    """Validate a value against an enum constraint."""
    errors = []
    if "enum" in prop_schema:
        allowed = prop_schema["enum"]
        if value not in allowed:
            errors.append(f"field '{field_name}': value '{value}' not in {allowed}")
    return errors


def validate_type(value, prop_schema: dict, field_name: str) -> list[str]:
    """Validate a value's type against the schema type constraint."""
    errors = []
    schema_type = prop_schema.get("type")
    if schema_type is None:
        return errors

    type_list = schema_type if isinstance(schema_type, list) else [schema_type]
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }

    ok = False
    for t in type_list:
        expected = type_map.get(t)
        if expected and isinstance(value, expected):
            ok = True
            break
        if t == "null" and value is None:
            ok = True
            break

    if not ok:
        errors.append(
            f"field '{field_name}': expected type {type_list}, got {type(value).__name__}"
        )
    return errors


def validate_artifact(artifact: dict, schema: dict) -> list[str]:
    """Validate an artifact dict against a JSON schema (lightweight, no external deps)."""
    errors = []

    # Check required fields
    errors.extend(validate_required(artifact, schema))

    # Check each field present in the artifact
    properties = schema.get("properties", {})
    for field, value in artifact.items():
        if field not in properties:
            if schema.get("additionalProperties") is False:
                errors.append(f"unexpected field: {field}")
            continue

        prop_schema = properties[field]
        errors.extend(validate_type(value, prop_schema, field))

        if value is not None:
            errors.extend(validate_enum(value, prop_schema, field))

            # Check pattern for strings
            if isinstance(value, str) and "pattern" in prop_schema:
                import re
                if not re.match(prop_schema["pattern"], value):
                    errors.append(
                        f"field '{field}': value '{value}' does not match pattern '{prop_schema['pattern']}'"
                    )

            # Check minimum for numbers
            if isinstance(value, (int, float)) and "minimum" in prop_schema:
                if value < prop_schema["minimum"]:
                    errors.append(
                        f"field '{field}': value {value} < minimum {prop_schema['minimum']}"
                    )

            # Check array items type
            if isinstance(value, list) and "items" in prop_schema:
                item_schema = prop_schema["items"]
                for i, item in enumerate(value):
                    errors.extend(validate_type(item, item_schema, f"{field}[{i}]"))

    return errors


def validate_file(artifact_path: Path, schema_name: str | None = None) -> tuple[bool, list[str]]:
    """Validate a single artifact file. Returns (ok, errors)."""
    if not artifact_path.is_file():
        return False, [f"file not found: {artifact_path}"]

    try:
        artifact = json.loads(artifact_path.read_text())
    except json.JSONDecodeError as e:
        return False, [f"invalid JSON: {e}"]

    if schema_name is None:
        schema_name = guess_schema_name(artifact_path)
    if schema_name is None:
        return False, [f"cannot determine schema for {artifact_path}"]

    try:
        schema = load_schema(schema_name)
    except FileNotFoundError as e:
        return False, [str(e)]

    errors = validate_artifact(artifact, schema)
    return len(errors) == 0, errors


def validate_all() -> dict[str, list[str]]:
    """Validate all known artifacts in the repo."""
    results = {}
    for dir_prefix, schema_name in DIR_SCHEMA_MAP.items():
        dir_path = REPO_DIR / dir_prefix
        if not dir_path.is_dir():
            continue
        for f in sorted(dir_path.glob("*.json")):
            ok, errors = validate_file(f, schema_name)
            results[str(f.relative_to(REPO_DIR))] = errors
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate JSON artifacts against schemas")
    parser.add_argument("files", nargs="*", help="Artifact files to validate")
    parser.add_argument(
        "--schema",
        choices=list(SCHEMA_MAP.keys()),
        default=None,
        help="Force a specific schema (otherwise auto-detected from path)",
    )
    parser.add_argument("--all", action="store_true", help="Validate all known artifacts")
    parser.add_argument("--quiet", action="store_true", help="Only print errors")
    args = parser.parse_args()

    if args.all:
        results = validate_all()
        if not results:
            print("No artifacts found to validate.")
            return

        any_error = False
        for path, errors in results.items():
            if errors:
                any_error = True
                print(f"FAIL  {path}")
                for e in errors:
                    print(f"      {e}")
            elif not args.quiet:
                print(f"OK    {path}")

        sys.exit(1 if any_error else 0)

    if not args.files:
        parser.print_help()
        sys.exit(1)

    any_error = False
    for filepath in args.files:
        path = Path(filepath)
        if not path.is_absolute():
            path = REPO_DIR / path
        ok, errors = validate_file(path, args.schema)
        if errors:
            any_error = True
            print(f"FAIL  {filepath}")
            for e in errors:
                print(f"      {e}")
        elif not args.quiet:
            print(f"OK    {filepath}")

    sys.exit(1 if any_error else 0)


if __name__ == "__main__":
    main()
