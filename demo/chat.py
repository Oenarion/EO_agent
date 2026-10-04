"""Talk to the agent from the terminal, against the running API. One session, as many turns as you like.

    python demo/chat.py                     # new session
    python demo/chat.py --session mytest    # resume or name a session

Commands:  /language  show or change the language of place names   /trace  print the trace of this session
           /memory    show the working memory and summary          /new    start a new session
           /quit      leave
"""
import argparse
import json
import sys
import time

import httpx

from eo_agent.config import DEFAULT_PLACE_LANGUAGE, LANGUAGE_NAMES, PLACE_LANGUAGES
from eo_agent.observability.pretty import render

INTRO = """\
EO scene agent
I find Sentinel-2 satellite scenes. Tell me a place and a period, and a cloud cover limit if you want one, for example:
    Find scenes over Ravenna, Italy in July 2025 with less than 10% cloud cover.
Then ask about the results, for example "give me the details of the second one".

What I can do:
  - find a place and its coordinates
  - search scenes for a place and dates, optionally with a cloud cover range
  - give the details of a scene (time, cloud cover, satellite, sun elevation, bands)
Every scene I mention comes from the catalogue and is listed with a link to its record.

Place names are searched in {language}. {hint}
Commands: /language  /trace  /memory  /new  /quit
"""


def language_line(code: str) -> str:
    return f"{LANGUAGE_NAMES[code]} ({code})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--session", default=f"chat-{int(time.time())}")
    args = parser.parse_args()
    session = args.session
    language = DEFAULT_PLACE_LANGUAGE
    options = ", ".join(f"{code} ({LANGUAGE_NAMES[code]})" for code in PLACE_LANGUAGES)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    with httpx.Client(base_url=args.api, timeout=300) as client:
        try:
            health = client.get("/health").json()
        except httpx.HTTPError as exc:
            sys.exit(f"Cannot reach the API at {args.api} ({type(exc).__name__}). Start it first, see the README.")
        print(INTRO.format(language=language_line(language), hint=f"Change it with /language <code>: {options}."))
        print(f"(model: {health['model']}, MCP tools loaded: {health['mcp_connected']}, session: {session})\n")

        while True:
            try:
                message = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not message:
                continue
            command = message.split()[0]
            if command == "/quit":
                break
            if command == "/language":
                parts = message.split()
                if len(parts) == 1:
                    print(f"Place names are searched in {language_line(language)}. Change it with /language <code>: {options}.\n")
                elif parts[1] in PLACE_LANGUAGES:
                    language = parts[1]
                    print(f"Place names will be searched in {language_line(language)} from the next message.\n")
                else:
                    print(f"Unknown language '{parts[1]}'. Choose one of: {options}.\n")
            elif command == "/new":
                session = f"chat-{int(time.time())}"
                print(f"New session: {session} (place name language stays {language_line(language)})")
            elif command == "/trace":
                r = client.get(f"/traces/{session}")
                print(render(r.json()["events"]) if r.status_code == 200 else "No trace yet.")
            elif command == "/memory":
                r = client.get(f"/sessions/{session}/memory")
                print(json.dumps(r.json(), indent=1, ensure_ascii=False) if r.status_code == 200 else "No such session yet.")
            else:
                r = client.post("/chat", json={"session_id": session, "message": message, "language": language})
                if r.status_code != 200:
                    print(f"[HTTP {r.status_code}] {r.json().get('detail', r.text)}\n")
                    continue
                body = r.json()
                for call in body["tool_calls"]:
                    print(f"  [tool] {call['tool']}({json.dumps(call['args'], ensure_ascii=False)}) -> {'ok' if call['ok'] else 'FAILED'}")
                print(f"agent> {body['reply']}\n")


if __name__ == "__main__":
    main()
