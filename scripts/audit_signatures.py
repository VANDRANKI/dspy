#!/usr/bin/env python3
"""Audit DSPy Signature subclasses for well-formed field descriptions.

Imports dspy, discovers Signature subclasses in the current working
directory (or a specified module), and checks that every InputField
and OutputField has a non-empty 'desc' string. Missing descriptions
result in weak prompts and confuse automatic optimizers.

Usage:
    python scripts/audit_signatures.py
    python scripts/audit_signatures.py --module my_program
    python scripts/audit_signatures.py --verbose
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path


def load_dspy():
    """Import dspy and return the module."""
    try:
        import dspy
        return dspy
    except ImportError:
        print("ERROR: dspy is not installed. Run: pip install dspy", file=sys.stderr)
        sys.exit(1)


def discover_signatures(dspy_module, target_module=None):
    """Return all non-abstract Signature subclasses found."""
    Signature = dspy_module.Signature
    source = target_module if target_module is not None else dspy_module
    subclasses = []
    for name, obj in inspect.getmembers(source, inspect.isclass):
        if (
            obj is not Signature
            and issubclass(obj, Signature)
            and not inspect.isabstract(obj)
        ):
            subclasses.append((name, obj))
    return subclasses


def audit_signature(name: str, cls, dspy_module) -> list[str]:
    """Return list of field issues in the Signature class."""
    issues: list[str] = []
    fields = getattr(cls, "fields", {})
    for field_name, field in fields.items():
        desc = getattr(field, "json_schema_extra", {}).get("desc", "") or ""
        if not desc.strip():
            issues.append(f"field '{field_name}' has no description")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module", metavar="NAME",
        help="Python module to inspect for Signature subclasses",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    dspy = load_dspy()

    target_module = None
    if args.module:
        try:
            target_module = importlib.import_module(args.module)
        except ImportError as exc:
            print(f"ERROR: Could not import '{args.module}': {exc}", file=sys.stderr)
            return 1

    signatures = discover_signatures(dspy, target_module)
    if not signatures:
        print("No Signature subclasses found.")
        return 0

    all_issues: dict[str, list[str]] = {}
    for name, cls in sorted(signatures):
        issues = audit_signature(name, cls, dspy)
        if issues:
            all_issues[name] = issues
        elif args.verbose:
            print(f"OK  {name}")

    if all_issues:
        print(f"\nSignature issues in {len(all_issues)}/{len(signatures)} class(es):\n")
        for sig_name, issues in sorted(all_issues.items()):
            for issue in issues:
                print(f"  {sig_name}: {issue}")
        return 1

    print(f"All {len(signatures)} Signature class(es) passed the audit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
