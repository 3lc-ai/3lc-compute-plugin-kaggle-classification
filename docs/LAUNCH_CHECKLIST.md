# LAUNCH_CHECKLIST.md: launch day for the Kaggle Classification plugin

The checks that can only run once the competition is live on Kaggle. Before launch, Kaggle's
submission list answers 403 (seen 2026-10-12 in the hardening campaign, docs/TEST_PLAN.md F4), so
everything that reads it is unexercised until the day the competition opens. Run these in order on a
tester environment with the release build installed from the hosted catalog, and record each result
with its date in TESTING_PROOF.md.

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
