#!/usr/bin/env python3
"""claude_onboard.py -- ONE-TIME setup of a new device from a USB/local payload.

The big raw transcripts (e.g. this 900 MB session) never go through git. Instead,
once per new machine, you carry them by hand and run this script. It:
  - copies all transcripts (.jsonl) from the payload into THIS machine's
    project-hash dir, so `claude --resume` can list them;
  - copies the memory/*.md into this machine's memory dir (re-homed across
    username/drive differences);
  - copies the global CLAUDE.md into this machine's ~/.claude.

After onboarding, day-to-day continuity is handled by /shutdown and /start (git).

Make a payload on the SOURCE machine (home):
  python scripts/claude_onboard.py export D:/claude_payload
    -> writes D:/claude_payload/{transcripts/*.jsonl, memory/*.md, CLAUDE.md}

Import on the NEW machine (work), from the USB path:
  python scripts/claude_onboard.py import D:/claude_payload
"""
import os, sys, glob, shutil

def claude_home():
    return os.environ.get("CLAUDE_CONFIG_DIR") or os.path.join(os.path.expanduser("~"), ".claude")

def projects_root():
    return os.path.join(claude_home(), "projects")

def project_hash_dir(create_if_missing=False):
    root = projects_root()
    os.makedirs(root, exist_ok=True)
    cands = [os.path.join(root, d) for d in os.listdir(root)
             if os.path.isdir(os.path.join(root, d))]
    withmem = [c for c in cands if os.path.isdir(os.path.join(c, "memory"))]
    pool = withmem or cands
    if pool:
        return max(pool, key=lambda p: os.path.getmtime(p))
    if create_if_missing:
        # No project dir yet on a fresh machine: create a placeholder. The real
        # hash dir is created by Claude Code on first launch; after that, re-run
        # import so transcripts land in the correct hash dir.
        d = os.path.join(root, "_onboard_pending")
        os.makedirs(os.path.join(d, "memory"), exist_ok=True)
        return d
    return None

def do_export(dest):
    pd = project_hash_dir()
    if not pd:
        sys.exit("No project-hash dir found on this machine; nothing to export.")
    os.makedirs(os.path.join(dest, "transcripts"), exist_ok=True)
    os.makedirs(os.path.join(dest, "memory"), exist_ok=True)
    n_t = n_m = 0
    for j in glob.glob(os.path.join(pd, "*.jsonl")):
        shutil.copy2(j, os.path.join(dest, "transcripts", os.path.basename(j))); n_t += 1
    md = os.path.join(pd, "memory")
    for m in glob.glob(os.path.join(md, "*.md")):
        shutil.copy2(m, os.path.join(dest, "memory", os.path.basename(m))); n_m += 1
    g = os.path.join(claude_home(), "CLAUDE.md")
    if os.path.isfile(g):
        shutil.copy2(g, os.path.join(dest, "CLAUDE.md"))
    print(f"Exported payload to {dest}: {n_t} transcripts, {n_m} memory files, "
          f"CLAUDE.md={'yes' if os.path.isfile(g) else 'no'}")
    print("Carry this folder to the new device (USB), then run: "
          "python scripts/claude_onboard.py import <payload_path>")

def do_import(src):
    if not os.path.isdir(src):
        sys.exit(f"Payload not found: {src}")
    pd = project_hash_dir(create_if_missing=True)
    md = os.path.join(pd, "memory"); os.makedirs(md, exist_ok=True)
    n_t = n_m = 0
    for j in glob.glob(os.path.join(src, "transcripts", "*.jsonl")):
        shutil.copy2(j, os.path.join(pd, os.path.basename(j))); n_t += 1
    for m in glob.glob(os.path.join(src, "memory", "*.md")):
        shutil.copy2(m, os.path.join(md, os.path.basename(m))); n_m += 1
    g = os.path.join(src, "CLAUDE.md")
    if os.path.isfile(g):
        shutil.copy2(g, os.path.join(claude_home(), "CLAUDE.md"))
    print(f"Imported into {pd}: {n_t} transcripts, {n_m} memory files, "
          f"CLAUDE.md={'yes' if os.path.isfile(g) else 'no'}")
    if os.path.basename(pd) == "_onboard_pending":
        print("\nNOTE: Claude Code had not created its project dir yet, so files went to\n"
              "  _onboard_pending. Launch Claude Code once in the project, then RE-RUN\n"
              "  this import so transcripts land in the real project-hash dir.")
    else:
        print("Done. `claude --resume` should now list the carried sessions.")

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("export", "import"):
        print(__doc__); sys.exit(1)
    (do_export if sys.argv[1] == "export" else do_import)(sys.argv[2])
