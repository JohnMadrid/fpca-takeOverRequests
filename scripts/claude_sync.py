#!/usr/bin/env python3
"""claude_sync.py -- engine for cross-device Claude Code continuity.

Carries WORKING continuity (not raw 900 MB chat logs) across machines via git:
  - memory .md files + global CLAUDE.md
  - LAST_SUMMARY.md (compaction summary, written by the /shutdown command)
  - the last N session transcripts, gzipped, SKIPPING any whose gzip > cap

Design decisions (see the /shutdown and /start commands):
  * Big raw transcripts are onboarded ONCE per device by hand (USB); see
    claude_onboard.py. Day-to-day, only the items above move through git.
  * "Last 3 sessions" are gzipped into claude-context/transcripts/. Any single
    session whose GZIPPED size exceeds SKIP_CAP_MB is SKIPPED and noted, so a
    mega-session never balloons the repo.
  * Paths differ across machines (different drive/username), so the project-hash
    transcript dir is resolved at runtime, and memory is re-homed on /start.

Usage:
  python claude_sync.py shutdown   # gather + commit + push (no auto-compact here)
  python claude_sync.py start      # pull + restore memory/CLAUDE.md to this machine
  python claude_sync.py status     # show what would be carried, sizes
"""
import os, sys, json, gzip, shutil, subprocess, hashlib, datetime, glob

# ---- config ----
LAST_N          = 3       # how many most-recent sessions to try to carry
SKIP_CAP_MB     = 150     # skip a session if its GZIP exceeds this many MB
PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTEXT_DIR     = os.path.join(PROJECT_ROOT, "claude-context")
CTX_MEM         = os.path.join(CONTEXT_DIR, "memory")
CTX_TRANS       = os.path.join(CONTEXT_DIR, "transcripts")
STATE_FILE      = os.path.join(CONTEXT_DIR, "SESSION_STATE.json")

# image workflow: session images stage here (gitignored), approved ones move to
# plots/keep (committed). Warn when plots/keep exceeds PLOTS_WARN_MB.
IMG_STAGE       = os.path.join(CONTEXT_DIR, "session-images")
PLOTS_KEEP      = os.path.join(PROJECT_ROOT, "plots", "keep")
PLOTS_WARN_MB   = 500
IMG_EXTS        = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf")

# ---- locate this machine's ~/.claude ----
def claude_home():
    # CLAUDE_CONFIG_DIR override, else ~/.claude
    return os.environ.get("CLAUDE_CONFIG_DIR") or os.path.join(os.path.expanduser("~"), ".claude")

def global_claude_md():
    return os.path.join(claude_home(), "CLAUDE.md")

def projects_root():
    return os.path.join(claude_home(), "projects")

def project_hash_dir():
    """The projects/<hash> dir for THIS machine's Claude home that holds the
    memory + transcripts. We pick the dir that actually contains a memory/ folder;
    if several exist, the most recently modified one wins."""
    root = projects_root()
    if not os.path.isdir(root):
        return None
    cands = [os.path.join(root, d) for d in os.listdir(root)
             if os.path.isdir(os.path.join(root, d))]
    withmem = [c for c in cands if os.path.isdir(os.path.join(c, "memory"))]
    pool = withmem or cands
    if not pool:
        return None
    return max(pool, key=lambda p: os.path.getmtime(p))

def mem_dir():
    d = project_hash_dir()
    return os.path.join(d, "memory") if d else None

# ---- helpers ----
def sh(args, cwd=PROJECT_ROOT, check=True):
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.stderr.write(f"$ {' '.join(args)}\n{r.stdout}{r.stderr}\n")
        raise SystemExit(r.returncode)
    return r

def mb(path):
    return os.path.getsize(path) / (1024*1024)

def gzip_file(src, dst):
    with open(src, "rb") as fi, gzip.open(dst, "wb", compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo)

def latest_transcripts(n):
    pd = project_hash_dir()
    if not pd:
        return []
    js = glob.glob(os.path.join(pd, "*.jsonl"))
    js.sort(key=os.path.getmtime, reverse=True)
    return js[:n]

def dir_mb(d):
    tot = 0
    for root, _, files in os.walk(d):
        for f in files:
            try: tot += os.path.getsize(os.path.join(root, f))
            except OSError: pass
    return tot / (1024*1024)

def staged_images():
    if not os.path.isdir(IMG_STAGE):
        return []
    return [os.path.join(IMG_STAGE, f) for f in os.listdir(IMG_STAGE)
            if f.lower().endswith(IMG_EXTS)]

# ============================================================ images
def list_images():
    """Print staged session images (for /shutdown to ask the user about)."""
    imgs = staged_images()
    if not imgs:
        print("No staged session images in claude-context/session-images/."); return
    print(f"{len(imgs)} staged session image(s):")
    for p in imgs:
        print(f"  {os.path.basename(p):50} {mb(p):7.2f} MB")

def keep_images():
    """Move ALL staged images into plots/keep/ (committed). Then size-warn."""
    os.makedirs(PLOTS_KEEP, exist_ok=True)
    imgs = staged_images(); moved = 0
    for p in imgs:
        dst = os.path.join(PLOTS_KEEP, os.path.basename(p))
        if os.path.exists(dst):  # avoid clobber: suffix with a counter
            base, ext = os.path.splitext(os.path.basename(p)); i = 1
            while os.path.exists(dst):
                dst = os.path.join(PLOTS_KEEP, f"{base}_{i}{ext}"); i += 1
        shutil.move(p, dst); moved += 1
    size = dir_mb(PLOTS_KEEP)
    print(f"Moved {moved} image(s) into plots/keep/  (now {size:.1f} MB)")
    if size > PLOTS_WARN_MB:
        print(f"  WARNING: plots/keep is {size:.0f} MB (> {PLOTS_WARN_MB} MB). "
              f"Sort through it and delete stale images to keep the repo lean.")

def discard_images():
    """Delete staged images without committing (user said no)."""
    imgs = staged_images()
    for p in imgs:
        os.remove(p)
    print(f"Discarded {len(imgs)} staged image(s) (not committed).")

def _oversized_tracked_or_new():
    """Return [(relpath, gz_mb), ...] for files git would commit whose GZIP size
    exceeds SKIP_CAP_MB. These stay USB-only and are summarised, not committed."""
    import tempfile
    r = sh(["git", "add", "-A", "--dry-run"], check=False).stdout
    paths = []
    for line in r.splitlines():
        line = line.strip()
        if line.startswith("add '") and line.endswith("'"):
            paths.append(line[len("add '"):-1])
    out = []
    for rel in paths:
        full = os.path.join(PROJECT_ROOT, rel)
        if not os.path.isfile(full):
            continue
        # cheap pre-filter: only gzip-test files whose RAW size could possibly
        # exceed the cap (gzip never grows data meaningfully for our content).
        if mb(full) <= SKIP_CAP_MB:
            continue
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".gz"); tmp.close()
        try:
            gzip_file(full, tmp.name); gz = mb(tmp.name)
        finally:
            os.remove(tmp.name)
        if gz > SKIP_CAP_MB:
            out.append((rel, gz))
    return out

# ============================================================ shutdown
def do_shutdown():
    os.makedirs(CTX_MEM, exist_ok=True)
    os.makedirs(CTX_TRANS, exist_ok=True)
    report = {"copied_memory": 0, "transcripts": [], "skipped": []}

    # 1) memory .md -> claude-context/memory
    md = mem_dir()
    if md and os.path.isdir(md):
        for f in glob.glob(os.path.join(md, "*.md")):
            shutil.copy2(f, os.path.join(CTX_MEM, os.path.basename(f)))
            report["copied_memory"] += 1

    # 2) global CLAUDE.md -> claude-context/CLAUDE.md
    g = global_claude_md()
    if os.path.isfile(g):
        shutil.copy2(g, os.path.join(CONTEXT_DIR, "CLAUDE.md"))

    # 3) last N transcripts -> gzip, skip if gz > cap
    # clear stale gz first so a removed session doesn't linger
    for old in glob.glob(os.path.join(CTX_TRANS, "*.jsonl.gz")):
        os.remove(old)
    for src in latest_transcripts(LAST_N):
        sid = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(CTX_TRANS, sid + ".jsonl.gz")
        gzip_file(src, dst)
        gz_mb = mb(dst)
        if gz_mb > SKIP_CAP_MB:
            os.remove(dst)
            report["skipped"].append({"session": sid, "gz_mb": round(gz_mb, 1)})
        else:
            report["transcripts"].append({"session": sid, "gz_mb": round(gz_mb, 1)})

    # 4) state file (records home project-hash dir name for relink on /start)
    pd = project_hash_dir()
    state = {
        "updated": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_project_dir": os.path.basename(pd) if pd else None,
        "last_n": LAST_N, "skip_cap_mb": SKIP_CAP_MB,
        "carried": report["transcripts"], "skipped": report["skipped"],
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    # 5) size guard: refuse to commit any file whose GZIP exceeds the cap.
    #    Such files stay USB-only and must be summarised in LAST_SUMMARY.md.
    #    (Notebooks gzip to tens of MB, well under the cap; this only catches
    #    genuine monsters like raw 900 MB transcripts.)
    oversized = _oversized_tracked_or_new()
    if oversized:
        report["oversized_skipped"] = oversized
        print("These files exceed the gzip cap and will NOT be committed "
              "(carry by USB, summarise in LAST_SUMMARY.md):")
        for f, gz in oversized:
            print(f"  {f}  ({gz:.0f} MB gz)")

    # 6) git add/commit/push -- stage everything except the oversized files
    sh(["git", "add", "-A", "claude-context"], check=True)
    sh(["git", "add", "-A"], check=True)
    for f, _gz in oversized:
        sh(["git", "reset", "-q", "HEAD", "--", f], check=False)
    date = datetime.date.today().isoformat()
    status = sh(["git", "status", "--porcelain"]).stdout.strip()
    if not status:
        print("Nothing to commit -- working tree and context already in sync.")
        _print_report(report); return
    msg = f"claude-sync {date}: memory({report['copied_memory']}) " \
          f"transcripts({len(report['transcripts'])}) skipped({len(report['skipped'])})"
    sh(["git", "commit", "-m", msg], check=True)
    push = sh(["git", "push"], check=False)
    if push.returncode != 0:
        print("Committed locally but PUSH FAILED:\n" + push.stderr)
    else:
        print("Pushed: " + msg)
    _print_report(report)

def _print_report(r):
    print(f"  memory files carried : {r['copied_memory']}")
    for t in r["transcripts"]:
        print(f"  transcript carried   : {t['session']}  ({t['gz_mb']} MB gz)")
    for s in r["skipped"]:
        print(f"  transcript SKIPPED   : {s['session']}  ({s['gz_mb']} MB gz > {SKIP_CAP_MB} MB cap)")
    if r["skipped"]:
        print("  -> skipped sessions are summarised in LAST_SUMMARY.md, not carried raw.")
    for f, gz in r.get("oversized_skipped", []):
        print(f"  file NOT committed   : {f}  ({gz:.0f} MB gz > {SKIP_CAP_MB} MB) -> USB + summarise")

# ============================================================ start
def do_start():
    # 1) pull
    pull = sh(["git", "pull", "--ff-only"], check=False)
    print(pull.stdout + pull.stderr)
    # 2) restore memory -> this machine's mem_dir (re-home across username/drive)
    md = mem_dir()
    restored = 0
    if md and os.path.isdir(CTX_MEM):
        os.makedirs(md, exist_ok=True)
        for f in glob.glob(os.path.join(CTX_MEM, "*.md")):
            shutil.copy2(f, os.path.join(md, os.path.basename(f)))
            restored += 1
    # 3) restore global CLAUDE.md
    ctx_cmd = os.path.join(CONTEXT_DIR, "CLAUDE.md")
    if os.path.isfile(ctx_cmd):
        os.makedirs(claude_home(), exist_ok=True)
        shutil.copy2(ctx_cmd, global_claude_md())
    # 4) relink carried transcripts into THIS machine's project-hash dir so
    #    `claude --resume` can list them (gunzip the .gz back to .jsonl).
    pd = project_hash_dir()
    relinked = []
    if pd and os.path.isdir(CTX_TRANS):
        for gz in glob.glob(os.path.join(CTX_TRANS, "*.jsonl.gz")):
            sid = os.path.basename(gz)[:-len(".jsonl.gz")]
            out = os.path.join(pd, sid + ".jsonl")
            if not os.path.exists(out):  # don't clobber a local copy
                with gzip.open(gz, "rb") as fi, open(out, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
            relinked.append(sid)
    print(f"Restored memory files: {restored}")
    print(f"Relinked transcripts : {len(relinked)} -> {', '.join(relinked) if relinked else '(none)'}")
    summ = os.path.join(CONTEXT_DIR, "LAST_SUMMARY.md")
    if os.path.isfile(summ):
        print(f"Last summary present : {summ}")
    else:
        print("No LAST_SUMMARY.md found (none written yet).")

# ============================================================ status
def do_status():
    print("Claude home      :", claude_home())
    print("Project-hash dir :", project_hash_dir())
    print("Memory dir       :", mem_dir())
    print("Context dir      :", CONTEXT_DIR)
    print(f"\nLast {LAST_N} transcripts (raw sizes):")
    for t in latest_transcripts(LAST_N):
        print(f"  {os.path.basename(t):40} {mb(t):8.1f} MB")
    print(f"\nWould skip any whose GZIP exceeds {SKIP_CAP_MB} MB.")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {
        "shutdown": do_shutdown, "start": do_start, "status": do_status,
        "list-images": list_images, "keep-images": keep_images,
        "discard-images": discard_images,
    }.get(cmd, do_status)()
