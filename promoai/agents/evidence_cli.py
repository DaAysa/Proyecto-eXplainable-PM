from __future__ import annotations

import argparse
import json
import os

from promoai.agents.evidence_sessions import check_saved_session
from promoai.general_utils.llm_connection import LLMConnection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PMAx EvidenceChecker on a persisted analysis session."
    )
    parser.add_argument("session_dir")
    parser.add_argument("--provider", default=os.getenv("PMAX_LLM_PROVIDER"))
    parser.add_argument("--model", default=os.getenv("PMAX_LLM_MODEL"))
    parser.add_argument("--request-index", type=int)
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Print the result without appending it to the saved session.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api_key = os.getenv("PMAX_API_KEY")
    if not api_key:
        raise SystemExit("PMAX_API_KEY is required.")
    if not args.provider or not args.model:
        raise SystemExit(
            "Provider and model are required via --provider/--model or "
            "PMAX_LLM_PROVIDER/PMAX_LLM_MODEL."
        )
    endpoint = os.getenv("PMAX_LLM_ENDPOINT")
    credentials = LLMConnection(
        api_key=api_key,
        llm_name=args.model,
        ai_provider=args.provider,
        args={"END_POINT": endpoint} if endpoint else {},
    )
    results = check_saved_session(
        args.session_dir,
        credentials,
        request_index=args.request_index,
        persist=not args.no_persist,
    )
    print(
        json.dumps(
            {str(index): result.as_dict() for index, result in results.items()},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

