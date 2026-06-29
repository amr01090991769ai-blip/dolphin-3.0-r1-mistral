#!/usr/bin/env python3
"""Sentinel — unified AI platform CLI.

Usage:
  python main.py serve [--host H] [--port P]      Start API + web dashboard
  python main.py agent "<goal>"                   Run the agent on a goal
  python main.py chat  "<prompt>"                 Single-turn chat
  python main.py scan  [path]                     Defensive security scan
  python main.py status                           Show platform status

Configuration is read from sentinel.json / config.json and SENTINEL_* env vars.
Set OPENAI_API_KEY (and optionally SENTINEL_OPENAI_BASE_URL + SENTINEL_MODEL)
to use a real LLM; otherwise the offline 'echo' backend keeps everything runnable.
"""
import argparse
import json
import sys

from sentinel.platform import Sentinel


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sentinel", description="Sentinel unified AI platform")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="Start API + web dashboard")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8080)

    p_agent = sub.add_parser("agent", help="Run the agent on a goal")
    p_agent.add_argument("goal", nargs="+")

    p_chat = sub.add_parser("chat", help="Single-turn chat")
    p_chat.add_argument("prompt", nargs="+")

    p_scan = sub.add_parser("scan", help="Defensive security scan")
    p_scan.add_argument("path", nargs="?", default=".")

    sub.add_parser("status", help="Show platform status")

    args = parser.parse_args(argv)
    s = Sentinel()

    if args.cmd == "serve":
        from sentinel.api.server import run_server
        httpd = run_server(args.host, args.port, sentinel=s)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[Sentinel] shutting down.")
            httpd.shutdown()
    elif args.cmd == "agent":
        goal = " ".join(args.goal)
        result = s.run_agent(goal)
        for i, step in enumerate(result["steps"], 1):
            print(f"\n--- Step {i} ---")
            if step["thought"]:
                print("Thought:", step["thought"])
            if step["action"]:
                print(f"Action: {step['action']}({step['action_input']})")
            if step["observation"]:
                print("Observation:", step["observation"])
        print("\n=== Final Answer ===")
        print(result["final_answer"])
    elif args.cmd == "chat":
        print(s.chat(" ".join(args.prompt)))
    elif args.cmd == "scan":
        print(s.scanner.scan_path(args.path).summary())
    elif args.cmd == "status":
        print(json.dumps(s.status(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
