"""Context policy demo: a long scripted conversation against the running API.

Start the API with LOW thresholds, so the policy triggers within 15 turns:
    MAX_WINDOW=4  SUMMARY_TRIGGER=8     (see the README for the PowerShell and bash forms)

For every turn it prints, taken from the trace:
  state    messages stored in the session at the start of the turn (with the new question)
  sent     messages sent to the model on its last call of the turn (system prompt not counted)
  chars    characters sent on that call (system prompt and memory block included)
  removed  messages folded into the summary during the turn (0 = no summarization)

The last question can only be answered from the summary: the working memory by then
holds a later search.

    python demo/run_context_demo.py --session context_policy
"""
import argparse
import sys
import time

import httpx

TURNS = [
    "Find Sentinel-2 scenes over Ravenna, Italy in July 2025 with less than 10% cloud cover.",
    "Give me the details of the second one.",
    "Thanks, that is useful.",
    "Now find scenes over Imola, Italy in September 2026 with more than 50% cloud cover.",
    "What is the cloud cover of the first one?",
    "What can you do for me?",
    "Find scenes over Venice, Italy in August 2025 with less than 5% cloud cover.",
    "Give me the details of the third one.",
    "Which satellite took it?",
    "Great. And how many scenes were found in total in that search?",
    "Find scenes over Bologna, Italy in June 2025 with less than 20% cloud cover, limit 3.",
    "Details of the last one, please.",
    "Hello again!",
    "Go back to the Imola search: how many scenes were found there?",
    "Going back to the very first search, over Ravenna: what was the cloud cover of the best scene?",
]


def turn_stats(events: list[dict], turn: int) -> dict:
    prepared = [e["data"] for e in events if e["turn"] == turn and e["event"] == "node" and e["node"] == "prepare_context" and e["data"]]
    if not prepared:
        return {}
    return {
        "state": prepared[0]["messages_in_state"],
        "sent": prepared[-1]["messages_sent"],
        "chars": prepared[-1]["chars_sent"],
        "removed": sum(p.get("removed", 0) for p in prepared),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--session", default=f"context-{int(time.time())}")
    args = parser.parse_args()

    with httpx.Client(base_url=args.api, timeout=300) as client:
        try:
            health = client.get("/health").json()
        except httpx.HTTPError as exc:
            sys.exit(f"Cannot reach the API at {args.api} ({type(exc).__name__}). Start it first, see the README.")
        policy = health["context_policy"]
        print(f"Model: {health['model']}. Policy in the server: window={policy['max_window']}, summary after >{policy['summary_trigger']} messages")
        if policy["summary_trigger"] > 10:
            print("WARNING: the thresholds are high, summarization may not trigger. Restart the API with low ones (see the README).")
        print(f"Session: {args.session}\n")
        print(f"{'turn':>4} {'state':>6} {'sent':>5} {'chars':>6} {'removed':>8}  question")
        replies = []
        for number, question in enumerate(TURNS, start=1):
            response = client.post("/chat", json={"session_id": args.session, "message": question})
            if response.status_code != 200:
                print(f"{number:>4}  HTTP {response.status_code}: {response.json().get('detail')}")
                replies.append("")
                continue
            replies.append(response.json()["reply"])
            events = client.get(f"/traces/{args.session}").json()["events"]
            s = turn_stats(events, number)
            print(f"{number:>4} {s.get('state', '?'):>6} {s.get('sent', '?'):>5} {s.get('chars', '?'):>6} {s.get('removed', '?'):>8}  {question[:70]}")

        print("\nLast question:", TURNS[-1])
        print("Answer:       ", replies[-1])
        memory = client.get(f"/sessions/{args.session}/memory").json()
        print(f"\nMessages stored at the end: {memory['messages_in_state']} (for {len(TURNS)} turns)")
        place = (memory["working_memory"].get("place") or {}).get("name")
        print(f"Working memory now points to the latest search ({place}). Summary of older turns:\n{memory['summary'] or '(none)'}")


if __name__ == "__main__":
    main()
