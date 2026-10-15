"""A record of every tool call made during an agent run.

The verifier needs the full results, not just what the model chose to print, so each tool wrapper
appends here. One agent runs at a time, so a module-level list is enough.
"""
LEDGER = []


def reset():
    LEDGER.clear()


def record(tool, args, result):
    LEDGER.append({"tool": tool, "args": args, "result": result})
    return result


def snapshot():
    return list(LEDGER)
