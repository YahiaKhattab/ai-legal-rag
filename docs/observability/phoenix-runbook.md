# Phoenix: Windows setup and first tracing increment

Base reviewed: `9846654`, main after chat-context merge. This is a development tracing increment, not a completed evaluation platform or a production deployment. Read [the revised plan](phoenix-plan.md) first.

## What is implemented

- Optional `phoenix` dependency extra: OpenTelemetry SDK and OTLP/HTTP exporter, both 1.44.0. No Phoenix server package or new ML framework inside the application environment.
- Metadata-only spans for CLI, API Ask, memory operations, query rewriting, pipeline initialization, embedding, dense retrieval, diversification, reranking, evidence assessment/selection, prompt construction, structured generation/retries and final result diagnostics.
- OpenInference kinds, actual gate thresholds, raw versus adjusted rerank scores, available Ollama token counts, purpose and retry ordinal. No embedding vectors, raw prompts, model answers, document text, names, keys, paths or thinking exported.
- API session correlation uses a keyed per-worker pseudonym, not the raw UUID. It resets after process restart and does not group sessions across workers. Existing Qdrant chat storage still contains application messages; this patch does not change that retention behavior.
- Tracing disabled by default. One owned provider per process, explicit development sampling of all spans, bounded background batch export (512 queued spans, batches of 64, 2-second request timeout), shutdown wait at most 3 seconds. Pending traces can be lost during outages or exit.
- Phoenix runs separately, pinned to release 20.11.0, with loopback port 6006 and persistent storage. No API Docker rebuild.
- Unconditional raw debug question/answer/HTTP-response dumps removed. CLI intentionally continues to print its question, answer and citations.

Remaining milestones: live Phoenix UI verification, richer per-validation-check spans, bounded synthetic content/evidence mapping, reviewed 40-example dataset and controlled experiments. Ingestion tracing and Grafana/Loki/etc. remain deferred. This initial metadata view is intentionally not a full textual RAG debugger.

## 1. Update the local repository without losing work

Use your normal PowerShell terminal. Keep an API server terminal separate.

```powershell
cd C:\Users\Yahia\source\repos\ai-legal-rag
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

function Invoke-GitChecked {
    & git @args
    if ($LASTEXITCODE -ne 0) { throw "Git command failed; stop and inspect the output." }
}

Invoke-GitChecked status --short --branch
Invoke-GitChecked fetch origin --prune

# Preserve tracked edits only. Large untracked archives stay where they are.
$trackedChanges = @(git status --porcelain --untracked-files=no)
if ($LASTEXITCODE -ne 0) { throw "Cannot inspect working tree." }
if ($trackedChanges.Count -gt 0) {
    Invoke-GitChecked stash push -m "Before Phoenix main update"
}

Invoke-GitChecked switch main
Invoke-GitChecked pull --ff-only origin main
Invoke-GitChecked log -1 --oneline
Invoke-GitChecked switch -c feature/phoenix-observability
```

If main has local commits and fast-forward fails, stop; do not reset or force it. If the branch already exists, inspect it instead of deleting it. If a stash was created, leave it preserved until you inspect `git stash list` and `git stash show --stat`; do not blindly pop configuration edits onto the new main.

Extract the supplied `phoenix-starter.zip` outside the repository, for example to `Downloads\phoenix-starter`. It contains `phoenix-observability.patch` plus these instructions and the plan. Set the path to where you actually extracted it:

```powershell
$patch = Join-Path $HOME "Downloads\phoenix-starter\phoenix-observability.patch"
if (-not (Test-Path -LiteralPath $patch)) { throw "Set `$patch to the downloaded patch file." }
Invoke-GitChecked apply --check $patch
Invoke-GitChecked apply $patch
Invoke-GitChecked diff --check
Invoke-GitChecked status --short --branch
```

If the patch check fails because the team has changed these files again, keep the worktree intact and provide the new commit/hash and error so the patch can be rebased. Do not use `git restore` to erase your own edits.

## 2. Dependencies and existing services

```powershell
python --version
python -m pip check
python -m pip install -e ".[phoenix]"
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
python -m pip check
```

Use Python 3.11 as specified by this repository. Do not recreate a working virtual environment. The core dependency list did not change in the reviewed team commits; an already up-to-date environment should mainly add the small tracing packages. If pip proposes unexpectedly large Torch/CUDA downloads, stop and inspect which existing requirements are missing before continuing. Do not add `[ocr]` for this work.

Start only the existing inference service if needed:

```powershell
docker compose up -d ollama
docker compose exec -T ollama ollama list
legal-rag-health
```

The current API uses `qwen3:4b` for answers and `qwen2.5:3b` for rewriting. If one is absent, pull only that model:

```powershell
# Run only for a missing model:
docker compose exec -T ollama ollama pull qwen2.5:3b
# docker compose exec -T ollama ollama pull qwen3:4b
```

Keep the configured Qdrant target. The host can use Cloud while the old API container uses local Qdrant; do not migrate data or rebuild containers to resolve this during tracing setup. Do not remove OpenEdx, Qdrant volumes, model volumes, or recovery backups.

## 3. Run focused tests before enabling tracing

```powershell
python -m pytest tests/test_observability.py -q
python -m ruff check src/legal_rag/observability tests/test_observability.py
python -m ruff format --check src/legal_rag/observability tests/test_observability.py
git diff --check
```

The new test module skips if the optional OTel SDK is absent. A skip is not a passed integration check: install the extra first.

Implementation verification performed in Python 3.11.16:

- Nine new tests passed using an in-memory exporter, mocked HTTP/model/database behavior and the real FastAPI route/pipeline boundaries.
- New module/test lint and formatting pass; focused type checking passes with imported modules followed silently and missing external types ignored. This is not a full-project mypy claim.
- Sixteen pre-existing query tests were compared on untouched main and the tracing branch with the same lightweight model stubs: both produced **5 passed, 11 failed**, with identical failing test names. Old tests still assume prior thresholds, `/api/generate` responses, or old CLI labels. Resolve those separately; do not claim a green full suite or reuse historical 101-pass results.
- The live Phoenix container, your Windows environment, real model generation and cloud corpus were not exercised from the authoring workspace. Follow the checks below to finish live validation.

## 4. Start Phoenix alone

```powershell
docker compose -p legal-rag-observability -f compose.phoenix.yml up -d phoenix
if ($LASTEXITCODE -ne 0) { throw "Phoenix startup failed." }
docker compose -p legal-rag-observability -f compose.phoenix.yml ps
Start-Process "http://127.0.0.1:6006"
```

This downloads the Phoenix image if absent, not a rebuilt RAG API image. It uses Docker's current disk location and a dedicated persistent volume. Release reference: [Phoenix 20.11.0](https://github.com/Arize-ai/phoenix/releases/tag/arize-phoenix-v20.11.0). Record the actual pulled image digest for repeatable experiments. Pull/start failure is a deployment problem, not evidence that the application patch failed.

Enable tracing for this terminal only:

```powershell
$env:LEGAL_RAG_PHOENIX_ENABLED = "true"
$env:LEGAL_RAG_PHOENIX_ENDPOINT = "http://127.0.0.1:6006/v1/traces"
$env:LEGAL_RAG_PHOENIX_PROJECT = "legal-rag-development"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
```

These settings are independent of existing Qdrant secrets. Do not paste `.env` or unrestricted environment dumps into reports. Do not name the Phoenix project after a banking client.

## 5. CLI smoke checks

Use an existing synthetic test collection/source if available. Do not infer labels from a different full cloud corpus. The following source is the historical synthetic demonstration; first confirm it exists in your current configured collection.

```powershell
$supportedQuery = 'وفقاً للمادة رقم ٣، ما الأحكام المتعلقة باستخدام بيانات المتعامل وعرض شروط ورسوم ومخاطر التعاقد الإلكتروني؟'
$unsupportedQuery = 'هل يحق للمتعامل إلغاء التعاقد الإلكتروني خلال أربعة عشر يوماً واسترداد جميع الرسوم دون إبداء أسباب؟'
legal-rag-query $supportedQuery --language ar --top-k 10 --top-n 6 --source "baseline-fintech"
legal-rag-query $unsupportedQuery --language ar --top-k 10 --top-n 6 --source "baseline-fintech"
```

Open project `legal-rag-development`. Verify one `cli.query` root per invocation, `rag.answer` beneath it, effective gate thresholds, candidate counts and final reason. A rejected request must have no final-answer `generation.attempt`. The supported query can still reveal a baseline gate problem; do not lower thresholds merely to make the screenshot pass. Phoenix observes decisions; it does not improve them by installation.

## 6. Current host API and chat

Start the latest code from the repository in a second activated terminal. Set the same three Phoenix variables **in that terminal** before launching:

```powershell
python -m uvicorn legal_rag_api.main:app --host 127.0.0.1 --port 8001
```

If port 8001 is already used by your earlier host API, stop that process with Ctrl+C in its own terminal and restart it so new imports/environment take effect. Leave the old Docker API on 8000 alone. Avoid `--reload` and multiple workers for the first trace check.

In the client terminal:

```powershell
Invoke-RestMethod "http://127.0.0.1:8001/health"
Start-Process "http://127.0.0.1:8001/docs"
```

Use `/legalAi/Ask` with a synthetic question. Copy the returned `session_id` into the next request and ask a follow-up. Unlike CLI retrieval, Ask writes messages to `chat_memory`; use a dedicated test Qdrant endpoint for an isolated chat experiment. No synthetic document upload or collection deletion is required by this tracing patch.

Check:

1. First turn: history read, contextualization span without rewrite LLM, then RAG and two memory writes.
2. Follow-up: history read, rewrite LLM under `query.contextualize`, then RAG. Check model and `rag.generation_purpose`.
3. Both requests have the same pseudonymous `session.id` on their `api.ask` roots within the same worker lifetime.
4. An unsupported follow-up may have a rewrite LLM call but must have no final-answer LLM call after gate rejection.
5. No questions, answers, legal text or raw session UUIDs appear in span attributes. API response shape remains unchanged.

If the Phoenix project is empty, check that the **server process** had tracing enabled before startup, inspect the standalone Phoenix container logs, and verify the endpoint ends `/v1/traces`. A healthy API alone is insufficient.

## 7. Disable, review and checkpoint

```powershell
$env:LEGAL_RAG_PHOENIX_ENABLED = "false"
# Restart the host API for this change to affect its process.
docker compose -p legal-rag-observability -f compose.phoenix.yml stop phoenix
git diff --check
git diff --stat
```

Stopping retains trace data. Do not run volume deletion as a routine shutdown step. Export can briefly log availability warnings while Phoenix is stopped, but legal requests must continue.

Once the focused tests and live traces are reviewed, commit only the intended application changes, tests, Compose file and docs. Do not add downloaded patches, archives, `.env`, trace exports or private evaluation data. Push the feature branch with `git push -u origin feature/phoenix-observability`; open a PR describing this as the first tracing increment, with the baseline failures and remaining evaluation work disclosed. Nothing in this bundle has already been pushed to GitHub for you.
