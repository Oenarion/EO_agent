"""Talk to the agent from the terminal, against the running API. One session, as many turns as you like.

    python demo/chat.py                     # new session
    python demo/chat.py --session mytest    # resume or name a session

Commands:  /trace  print the trace of this session   /memory  show the working memory and summary
           /new    start a new session               /quit    leave
"""
import argparse
import json
import sys
import time

import httpx

from eo_agent.observability.pretty import render


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--session", default=f"chat-{int(time.time())}")
    args = parser.parse_args()
    session = args.session
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    with httpx.Client(base_url=args.api, timeout=300) as client:
        try:
            health = client.get("/health").json()
        except httpx.HTTPError as exc:
            sys.exit(f"Cannot reach the API at {args.api} ({type(exc).__name__}). Start it first, see the README.")
        print(f"Model: {health['model']}. MCP tools loaded: {health['mcp_connected']}. Session: {session}")
        print("Type a message. Commands: /trace /memory /new /quit\n")

        while True:
            try:
                message = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not message:
                continue
            if message == "/quit":
                break
            if message == "/new":
                session = f"chat-{int(time.time())}"
                print(f"New session: {session}")
            elif message == "/trace":
                r = client.get(f"/traces/{session}")
                print(render(r.json()["events"]) if r.status_code == 200 else "No trace yet.")
            elif message == "/memory":
                r = client.get(f"/sessions/{session}/memory")
                print(json.dumps(r.json(), indent=1, ensure_ascii=False) if r.status_code == 200 else "No such session yet.")
            else:
                r = client.post("/chat", json={"session_id": session, "message": message})
                if r.status_code != 200:
                    print(f"[HTTP {r.status_code}] {r.json().get('detail', r.text)}\n")
                    continue
                body = r.json()
                for call in body["tool_calls"]:
                    print(f"  [tool] {call['tool']}({json.dumps(call['args'], ensure_ascii=False)}) -> {'ok' if call['ok'] else 'FAILED'}")
                print(f"agent> {body['reply']}\n")


if __name__ == "__main__":
    main()
