# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Section B of docs/TEST_PLAN.md (kit download and import) on a tester environment through the
in-process host. Each item is independent and records PASS / FAIL with its facts; the environment is
left with its kit re-downloaded and its tables in place (B7 adds initial-2 .. initial-4; B8 adds the
renamed project's import).

  <root>\\.venv\\Scripts\\python.exe tools/hardening/campaign_b.py --root <root> --repo <repo>
"""
from __future__ import annotations

import argparse
import json
import shutil
import socket
import sys
import time
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as h  # noqa: E402


def meta(client) -> dict:
    return client.get(f"{h.API}/config").json().get("_meta") or {}


def kit_dir_of(client) -> Path:
    ks = meta(client).get("kit_state") or {}
    return Path(ks.get("kit_dir") or "")


def closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--results", default=None)
    args = ap.parse_args()
    h.setup_stdio()
    root, repo = Path(args.root), Path(args.repo)
    results = h.Results(Path(args.results or (repo / "docs" / "campaign" / "section_b.json")))
    catalog = sorted(root.glob("catalog-campaign-*.json"))[-1]
    h.apply_env(h.env_for(root, catalog=str(catalog)))
    scratch = root / "campaign-scratch"
    scratch.mkdir(exist_ok=True)

    with h.host_client() as client:
        h.wait_ready(client, 900)
        m = meta(client)
        kit_dir = kit_dir_of(client)
        h.log(f"kit {m.get('kit_state')}; import {(m.get('import_state') or {}).get('state')}")

        # ── B6: wrong folder, nested folder, trailing slash ──────────────────
        try:
            wrong = scratch / "wrong-folder"
            wrong.mkdir(exist_ok=True)
            (wrong / "readme.txt").write_text("not a kit", encoding="utf-8")
            answers = {}
            for label, folder in (("wrong", str(wrong)), ("nested", str(kit_dir.parent)), ("trailing", str(kit_dir) + "\\"), ("quoted", f'"{kit_dir}"')):
                r = client.get(f"{h.API}/import/preflight", params={"kit_dir": folder})
                body = r.json()
                answers[label] = {"http": r.status_code, "all_ok": body.get("all_ok"), "error": (body.get("error") or "")[:160], "kit_dir": body.get("kit_dir")}
            ok = (answers["wrong"]["all_ok"] is False and answers["nested"]["all_ok"] is False
                  and answers["trailing"]["all_ok"] is True and answers["quoted"]["all_ok"] is True)
            results.put("B6", "PASS" if ok else "FAIL", answers=answers)
        except Exception as exc:
            results.put("B6", "FAIL", exception=f"{type(exc).__name__}: {exc}")

        # ── B3: a kit file modified after verify ─────────────────────────────
        try:
            v0 = client.get(f"{h.API}/download/verify").json()
            target = next(p for p in sorted((kit_dir / "data" / "train").rglob("*.jpg")))
            backup = scratch / "b3-backup.jpg"
            shutil.copy2(target, backup)
            with target.open("ab") as f:
                f.write(b"\x00tamper")
            v1 = client.get(f"{h.API}/download/verify").json()
            imp = h.run_and_wait(client, "import", {"mode": "reimport"}, timeout=1200, quiet=True)
            shutil.copy2(backup, target)
            v2 = client.get(f"{h.API}/download/verify").json()
            named = str(target.relative_to(kit_dir)).replace("\\", "/") in json.dumps(v1)
            results.put("B3", "PASS" if v0.get("ok") and v1.get("ok") is False and named and imp.get("status") == "failed" and v2.get("ok") else "FAIL",
                        before_ok=v0.get("ok"), after_ok=v1.get("ok"), mismatch_count=v1.get("mismatch_count"), names_the_file=named,
                        import_status=imp.get("status"), import_error=str(imp.get("error") or "")[:200], restored_ok=v2.get("ok"))
        except Exception as exc:
            results.put("B3", "FAIL", exception=f"{type(exc).__name__}: {exc}")

        # ── B7: re-import fresh three times ──────────────────────────────────
        try:
            names = []
            for _ in range(3):
                pre = client.get(f"{h.API}/import/preflight").json()
                expect = pre.get("reimport_name")
                job = h.run_and_wait(client, "import", {"mode": "reimport"}, timeout=1800, quiet=True)
                st = meta(client).get("import_state") or {}
                names.append({"expected": expect, "status": job.get("status"), "table": st.get("table_name"), "elapsed_s": job.get("elapsed_s")})
            got = [n["table"] for n in names]
            results.put("B7", "PASS" if all(n["status"] == "completed" and n["table"] == n["expected"] for n in names) and len(set(got)) == 3 else "FAIL", names=names)
        except Exception as exc:
            results.put("B7", "FAIL", exception=f"{type(exc).__name__}: {exc}")

        # ── B9: two imports started at once ──────────────────────────────────
        try:
            c1, b1 = h.start(client, "import", {"mode": "reimport"})
            c2, b2 = h.start(client, "import", {"mode": "reimport"})
            j1 = h.wait_terminal(client, b1.get("job_id") or b1.get("id"), timeout=1800, quiet=True) if c1 in (200, 202) else {"status": "refused"}
            j2 = h.wait_terminal(client, b2.get("job_id") or b2.get("id"), timeout=1800, quiet=True) if c2 in (200, 202) else {"status": "refused"}
            st = meta(client).get("import_state") or {}
            tables_dir = h.projects_dir(root) / st.get("project", "intel-scene") / "datasets" / "intel-scene_train" / "tables"
            partial = [p.name for p in tables_dir.iterdir() if p.is_dir() and not (p / "object.3lc.json").exists()] if tables_dir.is_dir() else []
            ok = {j1.get("status"), j2.get("status")} <= {"completed", "failed", "refused"} and not partial and st.get("state") == "success"
            results.put("B9", "PASS" if ok else "FAIL", first={"http": c1, "status": j1.get("status"), "error": str(j1.get("error") or b1.get("error") or "")[:160]},
                        second={"http": c2, "status": j2.get("status"), "error": str(j2.get("error") or b2.get("error") or "")[:160]},
                        partial_tables=partial, table_after=st.get("table_name"))
        except Exception as exc:
            results.put("B9", "FAIL", exception=f"{type(exc).__name__}: {exc}")

        # ── B1 / B2: cancel mid-shard, resume; a corrupted partial ───────────
        try:
            kit_version_dir = kit_dir.parent
            aside = scratch / "kit-aside"
            if aside.exists():
                shutil.rmtree(aside)
            shutil.move(str(kit_version_dir), str(aside))          # the kit is gone: a download is needed again
            # The dev CDN delivers the whole kit in about a second (section F: 111 MB/s), too fast to cancel
            # on a shard, so the interrupted state is built by hand from a download that kept its shards.
            first = h.run_and_wait(client, "download_kit", {"keep_archives": True}, timeout=1200, quiet=True)
            shards = sorted(p for p in kit_version_dir.glob("*.zip"))
            shutil.rmtree(kit_dir, ignore_errors=True)                 # the extracted tree is gone again
            truncated = shards[1]                                      # a shard cut in half becomes the .part the resume continues
            half = truncated.stat().st_size // 2
            data = truncated.read_bytes()[:half]
            truncated.unlink()
            truncated.with_suffix(".zip.part").write_bytes(data)
            corrupted = shards[2]                                      # a full shard with 64 bytes flipped: sha256 mismatch, re-download
            raw = bytearray(corrupted.read_bytes())
            raw[len(raw) // 2:len(raw) // 2 + 64] = b"\xff" * 64
            corrupted.write_bytes(bytes(raw))
            parts = sorted(p.name for p in kit_version_dir.rglob("*.part"))
            import hashlib
            bad_sha = hashlib.sha256(corrupted.read_bytes()).hexdigest()
            t0 = time.time()
            second = h.run_and_wait(client, "download_kit", {"keep_archives": True}, timeout=1200, quiet=True)
            resume_s = round(time.time() - t0, 1)
            ks = meta(client).get("kit_state") or {}
            part_gone = not truncated.with_suffix(".zip.part").exists() and truncated.exists()
            good_sha = hashlib.sha256(corrupted.read_bytes()).hexdigest() if corrupted.exists() else ""
            results.put("B1", "PASS" if first.get("status") == "completed" and parts and second.get("status") == "completed" and ks.get("state") == "success" and part_gone else "FAIL",
                        parts_before_resume=parts, second_status=second.get("status"), part_consumed=part_gone, resume_s=resume_s, kit_state=ks.get("state"),
                        note="the host's job list carries no log lines; the evidence is the .part consumed into the full shard and the kit verified")
            results.put("B2", "PASS" if second.get("status") == "completed" and good_sha and good_sha != bad_sha else "FAIL",
                        corrupted=corrupted.name, sha_before=bad_sha[:12], sha_after=good_sha[:12], note="the corrupted shard's bytes were replaced by a fresh download")
            for z in kit_version_dir.glob("*.zip"):
                z.unlink()
            if aside.exists():
                shutil.rmtree(aside)
        except Exception as exc:
            results.put("B1", "FAIL", exception=f"{type(exc).__name__}: {exc}")

        # ── B4: kit folder deleted after import ──────────────────────────────
        try:
            kit_dir = kit_dir_of(client)
            aside = scratch / "kit-aside-b4"
            if aside.exists():
                shutil.rmtree(aside)
            shutil.move(str(kit_dir.parent), str(aside))
            m4 = meta(client)
            pre = client.get(f"{h.API}/import/preflight").json()
            train = h.run_and_wait(client, "train", {"epochs": "1", "run_name": "campaign b4"}, timeout=600, quiet=True)
            doc = client.get(f"{h.API}/status/doctor?kaggle=0").json()
            images = (doc.get("import") or {}).get("images") or {}
            shutil.move(str(aside), str(kit_dir.parent))
            ok = ((m4.get("kit_state") or {}).get("state") == "stale" and pre.get("all_ok") is False
                  and train.get("status") in ("failed", "refused") and "not on this machine" in str(train.get("error") or train.get("body") or "")
                  and images.get("train", {}).get("state") == "missing")
            results.put("B4", "PASS" if ok else "FAIL", kit_state=(m4.get("kit_state") or {}).get("state"), preflight_error=str(pre.get("error") or "")[:160],
                        train_status=train.get("status"), train_error=str(train.get("error") or train.get("body") or "")[:220], doctor_images={k: v.get("state") for k, v in images.items()})
        except Exception as exc:
            results.put("B4", "FAIL", exception=f"{type(exc).__name__}: {exc}")

    # ── B5: a manual kit folder with the CDN blocked (a new host under a dead base URL) ──
    try:
        kit_copy = scratch / "manual-kit" / "starter_kit"
        if kit_copy.parent.exists():
            shutil.rmtree(kit_copy.parent)
        src = None
        with h.host_client() as client:
            h.wait_ready(client, 600)
            src = kit_dir_of(client)
        shutil.copytree(src, kit_copy)
        port = closed_port()
        h.apply_env({"KAGGLE_CLASSIFICATION_MANIFEST_BASE_URL": f"http://127.0.0.1:{port}"})
        with h.host_client() as client:
            h.wait_ready(client, 600)
            for _ in range(30):   # the background refresh must fail against the closed port first
                m5 = meta(client)
                if (m5.get("manifest_fallback") or {}).get("refresh_state") in ("done", "failed"):
                    break
                time.sleep(2)
            v = client.get(f"{h.API}/download/verify", params={"kit_dir": str(kit_copy)}).json()
            pre = client.get(f"{h.API}/import/preflight", params={"kit_dir": str(kit_copy)}).json()
            doc = client.get(f"{h.API}/status/doctor?kaggle=0").json()
            fb = (doc.get("manifest") or {}).get("fallback") or {}
            ks = meta(client).get("kit_state") or {}
            results.put("B5", "PASS" if (m5.get("manifest_fallback") or {}).get("active") and v.get("ok") and pre.get("all_ok") and ks.get("source") == "manual" else "FAIL",
                        fallback=m5.get("manifest_fallback"), verify_ok=v.get("ok"), verify_source=v.get("source"), preflight_ok=pre.get("all_ok"),
                        doctor_fallback=fb, kit_state_source=ks.get("source"))
        h.apply_env({"KAGGLE_CLASSIFICATION_MANIFEST_BASE_URL": h.MANIFEST_BASE})
        # Put the recorded kit back to the downloaded one (the manual record replaced it).
        with h.host_client() as client:
            h.wait_ready(client, 600)
            v = client.get(f"{h.API}/download/verify", params={"kit_dir": str(src)}).json()
            h.log(f"recorded kit restored to the download: ok={v.get('ok')} source={v.get('source')}")
    except Exception as exc:
        results.put("B5", "FAIL", exception=f"{type(exc).__name__}: {exc}")

    # ── B8: project renamed on disk ──────────────────────────────────────────
    try:
        with h.host_client() as client:
            h.wait_ready(client, 600)
            st0 = meta(client).get("import_state") or {}
            project = st0.get("project") or "intel-scene"
            pdir = h.projects_dir(root) / project
            renamed = pdir.with_name(project + "-renamed")
            if renamed.exists():
                shutil.rmtree(renamed)
            shutil.move(str(pdir), str(renamed))
            st1 = meta(client).get("import_state") or {}
            job = h.run_and_wait(client, "import", {"project_name": project + "-fresh", "mode": "import"}, timeout=1800, quiet=True)
            st2 = meta(client).get("import_state") or {}
            shutil.move(str(renamed), str(pdir))
            # Back on the original project: an import into it reuses its tables and the record names it again.
            back = h.run_and_wait(client, "import", {"project_name": project, "mode": "import"}, timeout=1800, quiet=True)
            st3 = meta(client).get("import_state") or {}
            results.put("B8", "PASS" if st1.get("state") == "stale" and sorted(st1.get("missing") or []) == ["train", "val"] and job.get("status") == "completed" and st2.get("project") == project + "-fresh" else "FAIL",
                        after_rename=st1.get("state"), missing=st1.get("missing"), import_into_new=job.get("status"), new_project=st2.get("project"),
                        after_restore=st3.get("state"), record_project_after=st3.get("project"), import_back=back.get("status"),
                        note="the renamed project's tables were imported under a new name; the record was then put back on the original project")
    except Exception as exc:
        results.put("B8", "FAIL", exception=f"{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
