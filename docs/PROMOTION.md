# PROMOTION.md — staging the competition objects on the CDN

Two tiers serve **byte-identical** objects (every URL inside them is relative), so promotion
is a pure copy of keys from the dev bucket to the prod bucket.

| Tier | Bucket | Served at | Who writes |
|---|---|---|---|
| dev | `3lc-competitions-dev` | `https://competitions.dev.3lc.ai` | Rishikesh, via the console |
| prod | (prod bucket) | `https://competitions.3lc.ai` | Gudbrand, by copying from dev after the plugin is complete |

Layout (`docs/PLAN.md` §A3), identical on both tiers:

| Key | Mutability | Content-Type | Cache-Control |
|---|---|---|---|
| `kaggle/classification-index.json` | **mutable** | `application/json` | `max-age=60` |
| `kaggle/intel-scene/manifest.json` | **mutable** (hotfixable) | `application/json` | `max-age=60` |
| `kaggle/intel-scene/starter-kit/v1/intel-scene-v1-NN.zip` | **immutable** once staged | `application/zip` | `public, max-age=31536000, immutable` |

CDN facts these rules rest on (verified on the dev distribution): Managed-CachingOptimized,
no origin request policy, object `Cache-Control` honored (min TTL 1 s), **query strings are
not part of the cache key** — so nothing in the plugin or these steps uses `?v=` busting, and
every mutable update is followed by an invalidation.

## 1. Produce the tree

From the plugin repo, with the kit already built (`tools/build_kit.py`):

```powershell
uv run python tools/make_cdn_tree.py --shards-dir "C:\Users\Owner\Desktop\3LC competitions\3LC Kaggle Competitions\datasets\intel-scene-kit-v1\shards" --out cdn --force
```

`cdn/` mirrors the bucket and `cdn/upload-plan.json` lists every key with its mutability,
Content-Type and Cache-Control. The tree is gitignored.

## 2. Upload to dev (console)

For every object in `upload-plan.json`, in the S3 console for `3lc-competitions-dev`:

1. **Upload** the local file to exactly the listed **key** (create the `kaggle/…` prefixes as
   folders; the key must match character for character).
2. **Before finishing the upload**, open **Properties → Metadata** and add two **System defined**
   entries: `Content-Type` = the listed value, `Cache-Control` = the listed value. Metadata is
   set **at upload time**: the Actions menu has no bulk metadata edit, so an object uploaded
   without them is fixed by **re-uploading** it with the metadata set.
3. Shards first, index last: the index is what makes the competition visible to plugins.

As done for the dev tier on 2026-09-28 (Rishikesh): **three separate uploads** — the five shards
under `kaggle/intel-scene/starter-kit/v1/`, then `kaggle/intel-scene/manifest.json`, then
`kaggle/classification-index.json` — each with the two metadata entries under Properties →
Metadata at upload time. The shards had to be uploaded a second time to carry the metadata the
first upload lacked (re-upload is the only fix); Last-Modified 15:28:47 GMT is that second pass.

Never overwrite an object under `starter-kit/v1/`. A changed kit is a new `kit.version`
(`starter-kit/v2/`) plus a manifest edit.

## 3. Invalidate after every mutable update (dev and prod alike)

After uploading or changing the index or a manifest, create a CloudFront invalidation on the
distribution serving that tier with exactly these paths:

```
/kaggle/classification-index.json
/kaggle/intel-scene/manifest.json
```

Without it a participant can see the old document for up to `max-age` (60 s) plus whatever
the edge already holds. Shards never need invalidation: their content never changes.

## 4. Verify what is actually served

```powershell
uv run python tools/verify_cdn.py https://competitions.dev.3lc.ai
```

The tool fetches index → manifest → every shard, checks each shard's sha256 and bytes against
the manifest, and prints the served `Content-Type`, `Cache-Control` and `X-Cache` per object.
It **fails** if a mutable object arrives without `max-age <= 300`, and warns if a shard lacks
`immutable`. Against a local `python -m http.server` (no headers) the header checks are skipped
and the report says so.

## 5. Promotion to prod (Gudbrand)

Copy these keys from `3lc-competitions-dev` to the prod bucket **with their metadata**
(`aws s3 cp --metadata-directive COPY`, or the console's copy which preserves metadata), then
invalidate the two mutable paths on the prod distribution and run the verification against
prod:

```powershell
uv run python tools/verify_cdn.py https://competitions.3lc.ai
```

Keys, sizes and sha256s of the current build (kit v1, 2026-09-22):

| Key | Bytes | sha256 |
|---|---|---|
| `kaggle/intel-scene/starter-kit/v1/intel-scene-v1-00.zip` | 547,781 | `70ca233f308f4bf80208bdef26f94b6eeac52e8e0a4c96c99b0f58d79f4c7072` |
| `kaggle/intel-scene/starter-kit/v1/intel-scene-v1-01.zip` | 21,235,172 | `ed72903f953a811203d1ddcc2cf7e26ffc1336a5272116f32b2e421d942b0073` |
| `kaggle/intel-scene/starter-kit/v1/intel-scene-v1-02.zip` | 53,237,641 | `908d7f91276a905b0d78960726c37a58d79d47878b000e83fdc04473ef1d3721` |
| `kaggle/intel-scene/starter-kit/v1/intel-scene-v1-03.zip` | 24,511,747 | `ba5c15cf7ef839ff745d37d36462d697cf4a9a6f9633f746958ab362c4003039` |
| `kaggle/intel-scene/starter-kit/v1/intel-scene-v1-04.zip` | 14,208,813 | `4e5ee9903863d10352b95a104c65891a254d3faa282e7bf0385a148b6068b0c2` |
| `kaggle/intel-scene/manifest.json` | 2,691 | `6a10e44f6d41157dc1148f914a1dc522701c3431c1c77637dfc8a609983efed2` |
| `kaggle/classification-index.json` | 203 | `e0e9c4bce89460607277fa7a57eead1d65c39be38a1d53d151a5eb63073f262d` |

The two mutable documents above are the **session 2.5 build** (display name "3LC Scene
Classification Challenge", `submission.daily_limit` 100 instead of the session-1 placeholder 3,
`cdn/` regenerated 2026-09-28, not yet uploaded). Until they are
re-uploaded and the two paths invalidated, the dev tier still serves the earlier documents
(manifest 2,699 B `2dd97576…4b38`, index 213 B `64c8fa92…41b0e`, verified in §6). The shards
are unchanged; both manifests describe the same kit v1.

Nothing on prod changes during session 2; this section is the hand-off for later.

## 6. Verified on dev (2026-09-28)

`tools/verify_cdn.py https://competitions.dev.3lc.ai` → **PASS: 5 shards, 113,741,154 bytes**.
Every sha256 in the table above is the value **served** by the dev distribution and equals the
manifest's `kit{}` block. Headers as served on the first fetch (`X-Cache: Miss from cloudfront`;
a re-fetch answered `Hit from cloudfront`):

| Object | Content-Type | Cache-Control | Last-Modified |
|---|---|---|---|
| `kaggle/classification-index.json` | `application/json` | `max-age=60` | Mon, 28 Sep 2026 15:36:59 GMT |
| `kaggle/intel-scene/manifest.json` | `application/json` | `max-age=60` | Mon, 28 Sep 2026 15:35:27 GMT |
| the five shards | `application/zip` | `public, max-age=31536000, immutable` | Mon, 28 Sep 2026 15:28:47 GMT |

The full Import ran on `3lc-hub-11` (compute 1.1.0, plugin installed from the test catalog pinned
to commit `4a42051`, `KAGGLE_CLASSIFICATION_MANIFEST_BASE_URL=https://competitions.dev.3lc.ai`):
manifest resolved `remote` from dev, download 68 s, import 18/18 checks, train 6,600 / val 1,200
rows. The promotion to prod (§5) is a copy of these seven keys with their metadata.

## What the plugin does with all this

The release build reaches `https://competitions.3lc.ai` only. Setting
`KAGGLE_CLASSIFICATION_MANIFEST_BASE_URL` (to the dev tier or a local `http://127.0.0.1:<port>`)
is the only way the plugin will talk to another host; `tests/test_manifest.py` fails the release
if the dev host is reachable without it. The plugin resolves the index, then the active
competition's manifest relative to the index, then each shard relative to the manifest — never
an absolute kit URL from the document — and verifies shards by sha256 from the manifest and
files by sha256 from `files.json`, never by ETag.
