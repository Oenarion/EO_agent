"""The demo: ONE session, three turns, against the running API.

  1. a question that needs tools (geocode, then search)
  2. a follow-up that only works if the context was kept ("the second one")
  3. a forced tool error (a scene id that does not exist)

Then it prints the trace of the session, so the three replies can be matched to it.

Start the MCP server and the API first (see the README), then:

    python demo/run_demo.py                      # session id demo-<time>
    python demo/run_demo.py --session error      # writes traces/error.jsonl
    python demo/run_demo.py --no-error --session normal   # only turns 1 and 2
"""
import argparse
import sys
import time

import httpx

from eo_agent.observability.pretty import render

TURNS = [
    "Find Sentinel-2 scenes over Ravenna, Italy in July 2025 with less than 10% cloud cover.",
    "Give me the details of the second one.",
    "Now give me the details of scene S2X_DOES_NOT_EXIST.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="base URL of the API")
    parser.add_argument("--session", default=f"demo-{int(time.time())}", help="session id (also the trace file name)")
    parser.add_argument("--no-error", action="store_true", help="skip the forced error turn")
    args = parser.parse_args()

    turns = TURNS[:2] if args.no_error else TURNS
    with httpx.Client(base_url=args.api, timeout=300) as client:
        try:
            health = client.get("/health").json()
        except httpx.HTTPError as exc:
            sys.exit(f"Cannot reach the API at {args.api} ({type(exc).__name__}). Start it first, see the README.")
        print(f"API ok. Model: {health['model']}. MCP connected: {health['mcp_connected']}. Session: {args.session}")

        for number, question in enumerate(turns, start=1):
            print(f"\n=== Turn {number}\nUser:  {question}")
            response = client.post("/chat", json={"session_id": args.session, "message": question})
            if response.status_code != 200:
                print(f"HTTP {response.status_code}: {response.json().get('detail')}")
                continue
            body = response.json()
            for call in body["tool_calls"]:
                print(f"  tool: {call['tool']}({call['args']})  ->  {'ok' if call['ok'] else 'FAILED'}")
            print(f"Agent: {body['reply']}")

        print("\n\n########## TRACE ##########")
        trace = client.get(f"/traces/{args.session}")
        if trace.status_code == 200:
            print(render(trace.json()["events"]))
        else:
            print("No trace found.")


if __name__ == "__main__":
    main()
