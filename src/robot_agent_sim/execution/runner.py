"""Small process boundary for MuJoCo control execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--viewer-mode", choices=("auto", "step", "headless"), default="auto")
    args = parser.parse_args()

    try:
        from .control_adapter import execute_bundle

        report = execute_bundle(args.bundle, viewer_mode=args.viewer_mode)
    except Exception as exc:  # the child must return a readable process error
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({
        "success": report.success,
        "commands_completed": report.commands_completed,
        "commands_total": report.commands_total,
        "runtime_steps": len(report.steps),
        "report": str(args.bundle.resolve().parent / "execution_report.json"),
    }, ensure_ascii=False))
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
