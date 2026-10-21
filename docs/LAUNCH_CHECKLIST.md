# LAUNCH_CHECKLIST.md: launch day for the Kaggle Classification plugin

The checks that can only run once the competition is live on Kaggle. Before launch, Kaggle's
submission list answers 403 (seen 2026-10-12 in the hardening campaign, docs/TEST_PLAN.md F4), so
everything that reads it is unexercised until the day the competition opens. Run these in order on a
tester environment with the release build installed from the hosted catalog, and record each result
with its date in TESTING_PROOF.md.

## Timeline (Rishikesh, 2026-10-21)

The hackathon is in the second week of November 2026. Targets: `1.0.0` cut by 2026-10-24, handed to
Gudbrand with docs/PROMOTION.md on 2026-10-27, prod verified (this checklist's pre-launch rows and the
Doctor on the production manifest) by 2026-10-31. The on-the-day rows below run when the competition opens.

## On the day the competition opens

- [ ] **Kaggle submission list check (organizer verification).** The anti-cheat review's vector 9
      (docs/PREDICT_MIRROR.md section 15) rests on comparing each team's bundle `ledger.jsonl` with
      Kaggle's own submission list for that team. With a connected token, open the Status tab: the
      "Kaggle live" section lists the account's submissions (no "available after the competition
      launches" note), and the History rows join to them by submission ref. Then export a verification
      bundle and confirm every `submit` entry's `kaggle.ref` in `ledger.jsonl` appears in Kaggle's list
      for the account, and that a submission made directly on kaggle.com (one, by hand) shows in Kaggle's
      list but has no ledger entry. That unmatched row is what organizers look for.
- [ ] **The Status tab's live section.** `GET /status/kaggle` answers `connected: true` with
      `submissions` and no `submissions_error`; the best public score and the rank are shown.
- [ ] **The leaderboard read.** The live section's board loads and the team's rank matches the
      Kaggle page.
- [ ] **The daily cap from Kaggle's side.** The connection card's "N of M left today" equals Kaggle's
      count for the account after one submission (the count is read from Kaggle's list, so it was 0 of
      100 before launch whatever had been submitted).
- [ ] **Not joined.** An account that has not accepted the competition rules gets the "competition not
      joined" card, and Submit is refused before any upload.
- [ ] **The manifest tier.** The production manifest base URL (docs/PROMOTION.md) serves the index and
      the manifest; the Doctor's Manifest row reads `remote` with the production URL.
- [ ] **The kit fallback on prod.** The served manifest is the regenerated 1.0.0 one: its `kit_fallback_url`
      is the Drive folder link (the check in docs/PROMOTION.md §5:
      `(Invoke-RestMethod https://competitions.3lc.ai/kaggle/intel-scene/manifest.json).kit_fallback_url` prints
      `https://drive.google.com/drive/folders/1mUleAk1PZtxyCooUVH-n6JfGEKn7Uc8N?usp=sharing`), and the Doctor's
      Kit fallback row shows the same link.
