#!/usr/bin/env python3
"""Assemble Checklist.md from the workflow output + hand-authored sections.

Reads the workflow .output JSON (285 domain items + 54 critique items),
merges hand-authored items for the 3 rate-limited domains (Model Hub, Datasets,
Auth/Security/Observability), de-duplicates by normalized title, groups by an
ordered area list, numbers items with stable area-codes, and writes Checklist.md.
"""
import html, json, re, sys
from collections import OrderedDict, Counter
from pathlib import Path

# Default input is the session workflow output (overridable via argv[1] or env).
# Checklist.md is the committed artifact; this script exists for provenance/re-runs.
_DEFAULT_OUT = r"C:/Users/lyani/AppData/Local/Temp/claude/C--Users-lyani-dFactory-Lab/41267a75-a055-4e84-92d8-6140ae86d1df/tasks/wc0bmw4to.output"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__import__("os").environ.get("DFACTORY_LAB_WF_OUTPUT", _DEFAULT_OUT))
DEST = Path(__file__).resolve().parent.parent / "Checklist.md"

# (code, area name) — defines section ORDER
AREA_ORDER = [
    ("F",  "Project Foundation, DevOps & Distribution"),
    ("B",  "Backend Core & Server Lifecycle"),
    ("R",  "Reliability, Orchestration & Unattended Operation"),
    ("MH", "Model Hub: Acquisition, Catalog & Cache"),
    ("M",  "MoE Weight Processing: Merge / Split / Modeling File"),
    ("D",  "Datasets & Training-Data Preparation"),
    ("T",  "Training Engine: dLLM Block-Diffusion SFT"),
    ("E",  "Export, Merge-Back & Packaging Pipeline"),
    ("I",  "Inference & Chat for dLLMs"),
    ("DL", "dLLM (Diffusion) Specific Capabilities & Eval"),
    ("H",  "Hardware Detection, GPU & VRAM Management"),
    ("A",  "Auth, Security & Observability"),
    ("U",  "Frontend App Shell & UX Foundation"),
    ("DS", "Design System & Visual Direction (distinct, non-slop)"),
    ("FE", "Frontend Feature Surfaces"),
]
AREA2CODE = {a: c for c, a in AREA_ORDER}
PHASE_NAMES = {
    0: "Foundation & Checklist",
    1: "Repo + backend skeleton + dev server",
    2: "Model hub + MoE merge/split",
    3: "Datasets & prep",
    4: "Training engine (dLLM SFT)",
    5: "Export/merge/convert + packaging",
    6: "Inference & chat",
    7: "Frontend shell + design system",
    8: "Frontend features",
    9: "Auth/security/observability/distribution",
    10: "Testing/QA/docs/release",
}

# ── Hand-authored items for the 3 rate-limited domains ─────────────────────
def a(area, cat, title, detail, layer, prio, phase):
    return {"area": area, "category": cat, "title": title, "detail": detail,
            "layer": layer, "priority": prio, "phase": phase}

MH = "Model Hub: Acquisition, Catalog & Cache"
DS_ = "Datasets & Training-Data Preparation"
AS = "Auth, Security & Observability"

MY_ITEMS = [
    # ── Model Hub ───────────────────────────────────────────────────────────
    a(MH, "Catalog", "Ship a dLLM model catalog seeded with LLaDA2.0-mini (16B) and LLaDA2.0-flash (100B)", "GET /api/models returns the supported models from the dFactory README (inclusionAI/LLaDA2.0-mini-preview, LLaDA2.0-flash-preview) with size, config path (configs/model_configs/llada2_mini, llada2_flash), and HF link.", "backend", "P0", 2),
    a(MH, "Catalog", "Parse each model's config.json to expose real metadata", "Read configs/model_configs/llada2_mini/config.json + llada2_flash/config.json; surface num_experts, hidden_size, num_hidden_layers, vocab_size and the mask_token_id/pad_token_id invariants to the UI.", "backend", "P1", 2),
    a(MH, "Download service", "Wrap snapshot_download as POST /api/models/download", "Server-side wrapper over scripts/download_hf_model.py (huggingface_hub.snapshot_download) launched as a managed subprocess job, not in-process, so a download crash cannot take the server down.", "backend", "P0", 2),
    a(MH, "Download service", "Return the TRUE resolved output path including the repo-name subfolder", "download_hf_model.py joins local_dir with repo_id.split('/')[1], so inclusionAI/LLaDA2.0-mini-preview into /x lands at /x/LLaDA2.0-mini-preview; the response must return that path or the next Merge step cannot find config.json.", "backend", "P0", 2),
    a(MH, "Download service", "Run downloads as a managed job with status queued/running/done/error/cancelled", "Persist job rows in SQLite; the job survives a UI reload and a server restart reconciles orphaned 'running' rows to 'error'.", "backend", "P0", 2),
    a(MH, "Transport & resume", "Offer HTTP and Xet transports with automatic downgrade", "Mirror Unsloth download_lifecycle: request Xet, fall back to HTTP when hf_xet is unavailable (resolve_effective_use_xet); reject with 400 when the chosen transport is unavailable.", "backend", "P1", 2),
    a(MH, "Transport & resume", "Make downloads resumable via a sequential HTTP writer", "Xet/hf_transfer write out-of-order so their partials cannot resume; default to the sequential HTTP writer so SIGKILL then resume is byte-identical for these large MoE snapshots.", "backend", "P1", 2),
    a(MH, "Transport & resume", "Stream download progress via SSE/WebSocket", "Emit {state, files_total, files_done, bytes_downloaded, bytes_total, eta} (Unsloth snapshot_progress / DownloadJobState); set HF_HUB_DISABLE_PROGRESS_BARS=1 and parse the HF callback rather than the TQDM bar.", "backend", "P0", 2),
    a(MH, "Transport & resume", "Support cancel (SIGTERM then SIGKILL), pause, and resume", "Cancel reaps the worker and leaves .incomplete partials for resume; protect blobs a concurrent same-repo peer is writing (protected_blob_hashes).", "backend", "P1", 2),
    a(MH, "HF token", "Persist an optional HF token for gated/large repos, never logged", "Store via providers storage (Unsloth storage/providers_db.py); pass as token= to snapshot_download; redact in all logs and API responses.", "backend", "P0", 9),
    a(MH, "HF token", "Validate the HF token (whoami) and surface validity in the UI", "GET /api/hf/validate calls whoami, returns the username and whether the token can read inclusionAI/LLaDA2.* ; mirror the use-hf-token-validation hook.", "fullstack", "P1", 2),
    a(MH, "Inventory & cache", "Build a local model inventory by scanning the cache + managed dir", "Mirror Unsloth inventory_scan/hf_cache_state + state_dir RepoType; GET /api/models/local lists downloaded models, detected format, size on disk, and completeness.", "backend", "P0", 2),
    a(MH, "Inventory & cache", "Detect whether a local model is separate-expert or merged-expert format", "Probe a MoE tensor name (experts.0.gate_proj vs experts.gate_proj) or shape [num_experts,...] to label the artifact and gate the Merge/Split actions in the UI.", "backend", "P1", 2),
    a(MH, "Inventory & cache", "Enforce disk-space pre-check before download", "Compare free space to the reported repo size (mini ~32GB+, flash ~200GB+); reject with a clear message rather than failing mid-download.", "backend", "P1", 2),
    a(MH, "Inventory & cache", "Expose a file manifest with blob hashes for integrity verification", "List snapshot files; on completion verify against HF-reported sha256 and store the hashes for later tamper checks.", "backend", "P1", 2),
    a(MH, "Local import & cleanup", "Allow importing an already-downloaded local model directory", "POST /api/models/import with a path; validate config.json + tokenizer + safetensors presence; register in inventory without re-downloading.", "backend", "P1", 2),
    a(MH, "Local import & cleanup", "Support model deletion that blocks on downstream run dependencies", "DELETE /api/models/{id} removes weights + cache entries; block and list referencing training runs if any exist.", "backend", "P2", 2),
    a(MH, "Catalog", "Allow downloading an arbitrary HF repo_id, not only the seeded catalog", "download_hf_model.py defaults to deepseek-ai/Janus-1.3B; support any repo_id but warn if the model is not a known dLLM (no merged-format guidance).", "backend", "P2", 2),
    a(MH, "Transport & resume", "Scrub the worker env for privacy and determinism", "Default HF_HUB_DISABLE_TELEMETRY=1 and HF_HUB_DISABLE_PROGRESS_BARS=1; do not forward HF_TOKEN into non-HF child processes (mirror Unsloth spawn_worker).", "backend", "P2", 2),
    a(MH, "UI", "Render the catalog as cards with format badge and gated actions", "Distinct (non-slop) card design; format badge = separate/merged; Download/Merge/Split buttons gated by the current format state.", "frontend", "P1", 8),
    a(MH, "UI", "Show a global download/job tray with pause/resume/cancel and per-file progress", "Driven by the job-framework progress stream; survives navigation.", "frontend", "P2", 8),

    # ── Datasets ────────────────────────────────────────────────────────────
    a(DS_, "Ingest", "Accept datasets from local upload, HF datasets, and paste-in text", "POST /api/datasets with multipart upload (jsonl/json/csv/parquet) or HF repo_id; mirror Unsloth routes/datasets.py + models/datasets.py.", "backend", "P0", 3),
    a(DS_, "Conversion", "Convert any tabular source into the conversational messages JSONL dFactory expects", "Target schema data_type:conversation, text_keys:messages -> [{role,content},...]; build_gsm8k_dataset.py is the reference (question/answer to messages).", "backend", "P0", 3),
    a(DS_, "Conversion", "Provide a configurable column-to-role mapper, not hardcoded question/answer", "build_gsm8k_dataset hardcodes question->user, answer->assistant; the UI must map arbitrary input columns to system/user/assistant roles and multi-turn arrays.", "fullstack", "P0", 3),
    a(DS_, "Build wrapper", "Wrap build_gsm8k_dataset.py as POST /api/datasets/build with overridable params", "The script hardcodes load_dataset('openai/gsm8k','main') and ./gsm8k_datasets; expose repo_id/config/output_dir as params, run as a managed job, return the absolute output path.", "backend", "P0", 3),
    a(DS_, "Build wrapper", "Pin the HF dataset revision and record its commit SHA in provenance", "build_gsm8k uses unpinned load_dataset; pin a revision and write {source, revision, built_at} as a sidecar so re-runs are byte-identical and staleness is detectable.", "backend", "P1", 3),
    a(DS_, "Build wrapper", "Emit the train.jsonl / test.jsonl split pair the trainer reads", "build_gsm8k writes gsm8k_{split}.jsonl per dataset split; the endpoint must emit train_path (and optional eval) and feed data.train_path in the YAML.", "backend", "P0", 3),
    a(DS_, "Validation", "Validate every row against the messages schema before training", "Ensure role in {system,user,assistant} and content is a non-empty string; reject/strip malformed rows with a count (oxc-validator-style fast path from Unsloth core/data_recipe).", "backend", "P0", 3),
    a(DS_, "Validation", "Warn when tokenized examples exceed data.max_seq_len and show the mask-boundary impact", "process_mdm_sft_example builds maskable_mask=arange(max_seq_len)>=prompt_length and labels[:prompt_length]=-100; truncation shifts the boundary. Show a length distribution with the model tokenizer.", "fullstack", "P1", 3),
    a(DS_, "Validation", "Detect a missing chat template and offer to attach the bundled LLaDA2 template", "train_llada2_bd.py raises 'No chat template found' if the tokenizer lacks chat_template; check after merge and offer the bundled template (assets/chat_templates) before train.", "fullstack", "P0", 3),
    a(DS_, "Preview & stats", "Render a row preview (raw + tokenized + applied chat template) and dataset stats", "Row count, avg/median/p95 token length, role distribution, dedupe count; mirror Unsloth data-recipe preview.", "fullstack", "P1", 3),
    a(DS_, "Preview & stats", "Apply the model chat template in the preview", "Use tokenizer.apply_chat_template on sample rows; show the templated string + token ids via the chat_templates resolver so users see exactly what trains.", "fullstack", "P1", 3),
    a(DS_, "Preview & stats", "Surface diffusion-specific data params (noise_range, block_size) as dataset-adjacent knobs", "These live in the YAML data/train sections but visually belong near the dataset; link them with inline help (noise_range = fraction of assistant tokens masked).", "fullstack", "P1", 4),
    a(DS_, "Storage & versioning", "Store built datasets under the managed data dir and version them", "./<data_dir>/datasets/<name>__<rev>; CRUD GET/DELETE /api/datasets; never commit datasets (gitignored).", "backend", "P1", 3),
    a(DS_, "Storage & versioning", "Record dataset-to-training-run lineage", "training_runs rows reference dataset_id + revision; enables reproducibility and a stale-dataset check.", "backend", "P1", 3),
    a(DS_, "Storage & versioning", "Stream large datasets without OOM (native dataloader)", "config dataloader_type:native; the builder handles streaming .jsonl for big sets.", "backend", "P2", 3),
    a(DS_, "Conversion", "Support multi-turn / sharegpt conversational input", "Accept [{role,content},...] arrays and messages-style JSON; map system/user/assistant turns.", "backend", "P2", 3),
    a(DS_, "Conversion", "Stream build progress (rows processed, percent) with cancel", "Jobs emit row counts; UI progress bar; cancel support.", "backend", "P1", 3),
    a(DS_, "Validation", "Deduplicate and optionally filter by length/quality", "Hash-based dedupe with count; min/max length filter; surface dropped counts in stats.", "backend", "P2", 3),
    a(DS_, "UI", "Build a dataset editor: source picker, column mapper, split controls, preview, build", "Distinct design; maps to POST /api/datasets/build; shows progress + resulting path with a 'Use in training' action.", "frontend", "P1", 8),
    a(DS_, "UI", "Offer the GSM8K final-answer marker strip/keep toggle", "gsm8k answer contains '... #### 42'; default verbatim to match the script and document the choice.", "fullstack", "P2", 3),

    # ── Auth, Security & Observability ──────────────────────────────────────
    a(AS, "Local auth", "Bootstrap a first-run access flow: printed token/URL + optional admin password", "Generate a bootstrap token at startup (startup banner), seed an admin user, allow first-login password change; study Unsloth auth/authentication.py + storage.py.", "backend", "P0", 9),
    a(AS, "Local auth", "Issue HS256 JWT access tokens (60min) + opaque refresh tokens (7d), secret in SQLite", "Replicate create_access_token/create_refresh_token/refresh_access_token; refresh tokens are opaque, hashed at rest, and not consumed on refresh.", "backend", "P1", 9),
    a(AS, "Local auth", "Hash passwords with PBKDF2-HMAC-SHA256 (100k iters) + per-user salt, constant-time verify", "Replicate auth/hashing.py (hash_password/verify_password); never store plaintext; use hmac.compare_digest.", "backend", "P1", 9),
    a(AS, "Local auth", "Distinguish desktop (loopback) vs web (exposed) sessions", "Unsloth marks desktop JWTs (to_encode['desktop']=True) and supports API_KEY_PREFIX api keys; the Lab needs both modes.", "backend", "P1", 9),
    a(AS, "Network exposure", "Bind to 127.0.0.1 by default; LAN/tunnel exposure is explicit opt-in", "Default DFACTORY_LAB_HOST=127.0.0.1; 0.0.0.0/--secure/tunnel require the bootstrap token and a changed password.", "backend", "P0", 9),
    a(AS, "Network exposure", "Auto-shutdown an exposed, unchanged-password instance after a deadline", "Replicate auth/bootstrap_timeout.py (default 3600s, env-driven, disabled on loopback/api-only); a fresh public instance with the default password is torn down.", "backend", "P1", 9),
    a(AS, "Network exposure", "Optional Cloudflare/LAN tunnel behind an explicit flag + token", "Study cloudflare_tunnel.py + run.py external-IP resolution; never auto-expose; require the access token to reach the tunneled URL.", "backend", "P2", 9),
    a(AS, "Credentials", "Store HF tokens / provider creds encrypted at rest, never logged", "Providers storage (providers_db.py); redact in logs and /api responses; per-credential ACL.", "backend", "P0", 9),
    a(AS, "Credentials", "Never write secrets, tokens, or weights to logs, telemetry, or error responses", "Log redaction filter + response sanitizer; CI asserts no token patterns appear in sample logs.", "backend", "P0", 9),
    a(AS, "Subprocess safety", "Run every wrapped CLI with argument arrays, never shell=True", "torchrun/moe_convertor/build_gsm8k must be subprocess([...], no shell); eliminates injection from user paths/repo_ids.", "backend", "P0", 9),
    a(AS, "Subprocess safety", "Jail all user-supplied paths under the managed data dir; reject traversal", "Normalize and verify every model_path/train_path/output_dir resolves under DFACTORY_LAB_DATA_DIR (or allow-listed roots); block ../ escapes.", "backend", "P0", 9),
    a(AS, "Subprocess safety", "Scrub/whitelist the child-process environment", "Mirror download_lifecycle spawn_worker env handling; set HF_HUB_DISABLE_* explicitly; do not forward HF_TOKEN into non-HF children.", "backend", "P1", 9),
    a(AS, "Subprocess safety", "Enforce per-job resource limits and graceful teardown", "Wall-clock + GPU-memory watchdog; kill runaway torchrun; on server shutdown send SIGTERM then SIGKILL and reap children.", "backend", "P1", 9),
    a(AS, "Observability", "Structured logging with request/job/run ids threaded through every line", "loggers/config.py + handlers.py; every line carries ids so a training run's logs are filterable.", "backend", "P1", 9),
    a(AS, "Observability", "Persist run/job history in SQLite", "studio_db-style tables; statuses queued/running/done/error/cancelled, started_at/ended_at, durations, exit codes, error tails.", "backend", "P1", 9),
    a(AS, "Observability", "Capture child-process exit codes + the failing torchrun rank in the run record", "Parse torchrun's per-rank summary; record failed_rank/exit_code so multi-GPU failures are diagnosable.", "backend", "P1", 9),
    a(AS, "Observability", "Live log viewer in the UI with rank filtering and search", "Stream captured stdout/stderr (rank-tagged) to a console pane; grep/filter; download full log.txt.", "frontend", "P1", 8),
    a(AS, "Observability", "Health/readiness endpoint reporting GPU, disk, and active jobs", "GET /api/health used by the UI connection-resilience banner and by load balancers; distinguishes up/busy/down.", "backend", "P1", 9),
    a(AS, "Audit", "Append-only audit log of model/data/credential operations", "audit_events table covers downloads, merges, deletes, token set/unset, training launches; exportable.", "backend", "P2", 9),
    a(AS, "Hardening", "Validate all request bodies with Pydantic; reject unknown fields", "Every route uses typed schemas (models/*.py); 422 on mismatch; no raw dict handling.", "backend", "P1", 9),
    a(AS, "Hardening", "Rate-limit auth endpoints (login/refresh) to blunt brute force", "Per-IP/per-subject throttling on /api/auth/*; lockout after N failures.", "backend", "P2", 9),
    a(AS, "Hardening", "CORS locked to the configured frontend origin; deny cross-origin by default", "Allow only desktop (tauri://) and the configured web origin; no wildcard in exposed mode.", "backend", "P1", 9),
    a(AS, "Hardening", "Scan dependencies for known vulns in CI", "pip-audit / npm audit / cargo audit weekly and on PR; fail on high CVEs in server/web/desktop.", "infra", "P2", 10),
]

def norm(t):
    return re.sub(r'[^a-z0-9]', '', t.lower())

def unescape(s):
    return html.unescape(s) if isinstance(s, str) else s

def main():
    raw = json.loads(OUT.read_text(encoding="utf-8"))
    result = raw.get("result", raw)
    items = list(result.get("items", [])) + list(result.get("extraItems", [])) + MY_ITEMS
    # unescape + normalize text fields
    for it in items:
        for k in ("area", "category", "title", "detail"):
            if k in it:
                it[k] = unescape(it[k])

    # de-dup by normalized title, keep first
    seen = set(); uniq = []
    for it in items:
        n = norm(it["title"])
        if n in seen:
            continue
        seen.add(n)
        uniq.append(it)

    # group by area (ordered), then by category (alpha)
    grouped = OrderedDict((a, OrderedDict()) for _, a in AREA_ORDER)
    unknown = []
    for it in uniq:
        area = it["area"]
        if area not in AREA2CODE:
            unknown.append(it); continue
        grouped[area].setdefault(it["category"], []).append(it)
    if unknown:
        print(f"WARN: {len(unknown)} items with unmapped area; first: {unknown[0]['area']!r}", file=sys.stderr)

    # assign codes + render
    by_phase = Counter(); by_prio = Counter(); by_layer = Counter(); by_area = Counter()
    lines_body = []
    for idx, (code, area) in enumerate(AREA_ORDER, 1):
        cats = grouped[area]
        area_count = sum(len(v) for v in cats.values())
        if area_count == 0:
            continue
        lines_body.append(f"\n## {idx}. {area}")
        lines_body.append(f"\n<sub>{area_count} items · section code `{code}`</sub>\n")
        n = 0
        for cat in sorted(cats.keys()):
            lines_body.append(f"**{cat}**\n")
            for it in cats[cat]:
                n += 1
                cid = f"{code}-{n}"
                by_phase[it["phase"]] += 1
                by_prio[it["priority"]] += 1
                by_layer[it["layer"]] += 1
                by_area[code] += 1
                title = it["title"].strip()
                detail = it["detail"].strip()
                lines_body.append(
                    f"- [ ] **{cid}** — {title}  "
                    f"`{it['priority']}` `ph{it['phase']}` `{it['layer']}`\n"
                    f"    - {detail}\n"
                )

    total = sum(by_phase.values())

    # ── stats table ────────────────────────────────────────────────────────
    stats = ["| Metric | Count |", "|---|---|", f"| **Total checkable items** | **{total}** |"]
    stats.append("\n**By phase**\n")
    stats.append("| Phase | Items |"); stats.append("|---|---|")
    for p in sorted(PHASE_NAMES):
        stats.append(f"| {p} — {PHASE_NAMES[p]} | {by_phase.get(p,0)} |")
    stats.append("\n**By priority**\n")
    stats.append("| Priority | Items |"); stats.append("|---|---|")
    for p in ("P0", "P1", "P2"):
        stats.append(f"| {p} | {by_prio.get(p,0)} |")
    stats.append("\n**By layer**\n")
    stats.append("| Layer | Items |"); stats.append("|---|---|")
    for l in ("backend", "frontend", "fullstack", "infra", "design"):
        stats.append(f"| {l} | {by_layer.get(l,0)} |")
    stats.append("\n**By section**\n")
    stats.append("| Code | Section | Items |"); stats.append("|---|---|---|")
    for code, area in AREA_ORDER:
        if by_area[code]:
            stats.append(f"| {code} | {area} | {by_area[code]} |")

    # ── assemble full document ─────────────────────────────────────────────
    header = HEADER.format(total=total)
    doc = header + "\n".join(stats) + "\n\n---\n" + "".join(lines_body) + "\n" + FOOTER
    DEST.write_text(doc, encoding="utf-8")
    print(f"Wrote {DEST} — {total} items "
          f"(workflow {len(result.get('items',[]))} + critique {len(result.get('extraItems',[]))} + authored {len(MY_ITEMS)}, "
          f"deduped from {len(items)})")

HEADER = """<!-- dFactory-Lab master checklist. Generated by scripts/build_checklist.py — re-run after editing item sources. -->

# dFactory-Lab — Master Checklist

> The single source of truth for building **dFactory-Lab**: a server + UI that wraps the
> [dFactory](https://github.com/inclusionAI/dFactory) diffusion-LLM (dLLM) fine-tuning engine so a user can
> **download → MoE-merge → prepare data → train (block-diffusion SFT) → split → export → chat** without touching the CLI.
> The fine-tuning/merge/diffusion logic comes from **dFactory (Apache-2.0)**; the **Unsloth Studio** architecture
> (FastAPI + Tauri/React) is the *management UI reference, studied but not copied* (it is AGPL-3.0).

**Status: Phase 0 — Foundation.** Every phase ends with a git commit. Items map 1:1 to PRs (`Closes §T-12`).

## How to use this checklist

- **Check boxes** — GitHub renders `- [ ]`; check an item when its acceptance criteria are met and verified.
- **Item codes** are stable per section (`MH-1`, `T-12`…); cite them in commits and PRs.
- **`P0`** blocks the MVP · **`P1`** core · **`P2`** polish.
- **`phN`** is the suggested build phase (table below). Layers: `backend` · `frontend` · `fullstack` · `infra` · `design`.
- Items are grounded in the real upstream source — file names, config keys (`block_diffusion_mode`, `noise_range_low/high`,
  `block_size`, `moe_implementation: fused`, `data_parallel_mode: fsdp2`), and exact script behavior
  (`scripts/moe_convertor.py`, `scripts/download_hf_model.py`, `scripts/build_gsm8k_dataset.py`, `train.sh`).

## Phase map

| ph | Phase | Repo state at end of phase |
|---|---|---|
| 0 | Foundation & Checklist | engine vendored, Checklist.md, project meta — **(this commit)** |
| 1 | Repo + backend skeleton + dev server | FastAPI app starts like Unsloth (run.py + banner), SQLite stores, job framework |
| 2 | Model hub + MoE merge/split | download, merge, split, inventory |
| 3 | Datasets & prep | ingest, conversational conversion, validation, preview |
| 4 | Training engine (dLLM SFT) | YAML builder, torchrun launcher, live metrics, resume |
| 5 | Export/merge/convert + packaging | automated split + modeling-file copy, packaging |
| 6 | Inference & chat | dLLM serving, masking/step params, history, compare |
| 7 | Frontend shell + design system | routing, layout, **locked distinct design system** |
| 8 | Frontend features | training/datasets/models/export/chat/playground screens |
| 9 | Auth/security/observability/distribution | auth, subprocess safety, audit, Tauri/web packaging |
| 10 | Testing/QA/docs/release | test suites, GPU smoke CI, docs site, v1 release |

## Design direction — LOCKED: (A) Editorial Paper

The UI **must be visually distinct** from Unsloth/dFactory and **must reject the generic "AI-slop" black + blue
gradient SaaS look.** Direction **locked 2026-07-01 = (A) Editorial Paper**; the other two are retained as fallbacks.

- **(A) Editorial Paper — LOCKED.** Warm off-white paper base, ink-near-black text, a single warm accent (terracotta/saffron),
  serif display + grotesque body + mono. Light-first, magazine-grade.
- **(B) Signal Lab** — warm near-black canvas, phosphor amber/green foreground, mono-dominant, oscilloscope/CRT feel.
  Dark, but amber/green (not blue).
- **(C) Industrial Console** — neutral light-gray base, high-contrast data density, a single hazard accent
  (safety-orange), mono readouts, instrument-panel layout.

## Counts

"""

FOOTER = """
---

### Notes & provenance

- **285** domain items + **54** critique items were authored by a multi-agent workflow that read the real dFactory and
  Unsloth Studio source (exact files, params, line-level behavior). Three domains that hit transient API rate limits
  (**Model Hub**, **Datasets**, **Auth/Security/Observability**) were authored by hand from the same source files.
  De-duplication is by normalized title; complementary items were retained.
- This file is **generated** by `scripts/build_checklist.py`. To revise the item set, edit the sources and re-run:
  `python scripts/build_checklist.py`.
- Engine provenance: `configs/`, `models/`, `scripts/`, `tasks/`, `train.sh` © inclusionAI (Apache-2.0). See
  [`LICENSE`](LICENSE), [`LEGAL.md`](LEGAL.md), [`docs/upstream/dFactory-README.md`](docs/upstream/dFactory-README.md).
"""

if __name__ == "__main__":
    main()
