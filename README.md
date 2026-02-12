A few bugs here and there, see repository's `flagship_prototype_general_alpha` branch for previous version before recent algorithm changing updates.

# Phragmén FIFO (Sequential) — quota floors, FIFO spend, audit

This repository implements an auditable **sequential Phragmén** engine for experiments with:

- **FIFO (time‑priority) credit spending** (cutoff time **τ**).
- **Quota‑floor ballots** (`electorate`, `party`, `mega`) that can be **active (unsatisfied)** or **dormant (satisfied)** each round.
- **Franchise / projection accounting** derived from base ballots (how much of the base franchise is “used” as winners are chosen).
- Parsing support for additional local ballot types (`partyrock`, `megarock`) that are **not yet fed into the main allocation** (parsing + audit only).

The code is intentionally bookkeeping‑heavy: most derived values are written to audit CSVs.

---

## Core concepts

### Base ballots

Base ballots are approval ballots (optionally aggregated). They are the “normal” voters in the sequential run.

### Quota‑floor ballots

Quota ballots represent **minimum quota floors**, not continuously‑available support balances.

At sequential round `r`:

- `required = ceil(quota_floor * r)`
- the quota ballot is **active** iff `winners_in_set < required`

Active quota ballots can contribute/spend under the configured spending tiers.
Dormant quota ballots accumulate reserve in FIFO time, but are not counted for selection pressure.

### Canonical share → quota_floor mapping

For any quota ballot we first compute a **share** `share = wr / N` (where `wr` is an absolute / derived weight and `N` is the arena population).

Then:

- `quota_floor = min((2/3) * share, 1/3)`
- `rel_weight = N * quota_floor`

This is the one definition used for `mega`, `party`, `electorate`, and in the PartyRock mini‑elections.

---

## Global election attributes

At JSON input (top‑level), provide:

- `total_population`  (wglobal1)
- `total_enrollment`  (wglobal2)
- `total_turnout`     (wglobal3)

Derived:

- `wglobal4 = max(wglobal3, sum(party wp1), sum(partyrock wpr1), sum(base wr))`
- `wglobal5 = max(wglobal2, sum(electorate we1))`

Notes:

- If a global is missing/invalid, the parser applies fallbacks (see `compute_global_totals`).
- The **arena N** used when converting to `rel_weight` is:
  - `N = wglobal4` for global quota ballots (`mega`, `party`, `electorate`)
  - `N = we3` for local quota ballots (`partyrock`, `megarock`)

---

## Ballot kinds and derived quantities

### Mega ballots (global quota)

Input per mega:

- `id`
- `weight` (w1)
- `candidates` (approvals)

Derived:

- `w2 = max(w1, sum(megarock wmr linked to this mega))`
- `share1 = w1 / wglobal1`
- `share2 = w2 / wglobal1`
- `share_used = max(share1, share2)`
- `quota_floor = min((2/3)*share_used, 1/3)`
- `rel_weight = wglobal4 * quota_floor`

### Electorate ballots (global quota)

Input per electorate:

- `id`
- `weight` (we1 = enrolled population)
- `turnout` (we2 = electorate turnout)
- `candidates` (may be extended by registered base/PartyRock ballots)

Derived:

- `we3 = max(we2, sum(base wr in electorate), sum(PartyRock wpr1 in electorate))`
- `share1 = we2 / wglobal5`
- `share2 = we3 / wglobal4`
- `share_used = max(share1, share2)`
- `quota_floor = min((2/3)*share_used, 1/3)`
- `rel_weight = wglobal4 * quota_floor`

Additional parsing‑time counters:

- `counted_base_ballots`, `counted_base_records`
- `counted_partyrock_abs_weight`, `counted_partyrock_records`, `counted_partyrock_ballots`

Auto‑creation:

- If base ballots or PartyRock ballots reference a missing electorate, a new electorate ballot is auto‑created.
- Auto‑created electorates are populated with candidates discovered from their associated ballots.

### Party ballots (global quota)

Input per party:

- `id`
- `weight` (wp1)
- `candidates` (ordered party list)

Derived:

- `wp2 = max(wp1, sum(PartyRock wpr1 linked to this party))`
- `wp3 = sum(PartyRock electorate‑normalised weights wpr2 linked to this party)`
- `share1 = wp2 / wglobal4`
- `share2 = wp3 / wglobal5`
- `share_used = max(share1, share2)`
- `quota_floor = min((2/3)*share_used, 1/3)`
- `rel_weight = wglobal4 * quota_floor`

Auto‑creation:

- If a PartyRock ballot references a missing party ballot, a placeholder party ballot is auto‑created.

### PartyRock ballots (local quota; parsing/audit only)

Input per PartyRock:

- `weight` (wpr1)
- `approvals` (ordered; used later for tie breaking / list extension)
- `electorate` reference
- `party` reference

Derived per associated electorate:

- `wpr2 = wpr1 * we1 / we3`
- `share1 = wpr1 / we2` (for future local quota use)
- Arena N for `rel_weight` conversion would be `we3`.

### MegaRock ballots (local quota; parsing/audit only)

Input per MegaRock:

- `weight` (wmr)
- `approvals`
- `electorate` reference
- `mega` reference

Derived:

- `share1 = wmr / (we1 * wglobal1 / wglobal5)`
- Arena N for `rel_weight` conversion is `we3`.

---

## Election Profiles

### Prototype flagship: `general_alpha`

Baseline FIFO sequential Phragmén with quota‑floor activation and the default spend tiers.

(Should credibly work ok for basic inquisitive experimental purposes, or low key entry level home administration renovation projects)

### `12` (currently under construction, yet to finish)

Adds constant (election‑wide) normalisation multipliers computed from absolute totals:

- `x = sum(base ballot weights)`
- `y = sum(party ballot weights)`
- `n = input (n/total_ballots) or defaults to max(x, y)`

Multipliers:

- Base voter multiplier: `(n + (2x - y)) / (3n)`
- Party multiplier: `(2n - (2x - y)) / (3n)`

### `324` (currently under construction, yet to finish)

Same as `12`, with different constants:

- Base voter multiplier: `(n + (2x - y))*2 / (9n)`
- Party multiplier: `(2n - (2x - y))*2 / (9n)`

And dynamic mega scaling per round (only for active mega quota groups):

- `z = sum(active mega rel_weights)`
- if `z <= n/3`: mega multiplier = `1`
- else: mega multiplier = `n / (3z)`

### Todo: More electoral topology profiles, e.g. tiered federal edition

---

## Party list extension from PartyRock ballots

If PartyRock ballots contain candidates not present in the party’s ordered list, those candidates are appended.

To generate an order for the appended segment, the tool runs a **mini `general_alpha` FIFO sequential election** for each party:

- **Candidates:** the party’s missing candidates.
- **Mini base ballots:** all base ballots whose approvals include any of those candidates (restricted to that candidate set).
- **Mini quota ballots:** the party’s PartyRock ballots (restricted to that candidate set).
- `mini_pop = max(sum(mini base abs), sum(mini PartyRock abs))`
- Mini quota mapping uses the same share→quota_floor mapping with `N = mini_pop`.

The resulting winner order is appended to the party list.

---

## Audit outputs

The CLI writes audit outputs to the output directory:

- `audit_globals.csv` — globals + (if relevant) profile 12/324 normalisation context.
- `audit_groups.csv` — one row per parsed group/ballot with meta JSON (includes derived shares, arena N, etc.).
- `audit_party_lists.csv` — party list ordering (after PartyRock extensions).

The sequential passes also write:

- `passXX_rounds.csv`
- `passXX_quota.csv`
- `passXX_projection.csv`

---

## CLI

Run an election:

```bash
python -m phragmen.cli run --input-json path/to/election.json --outdir out --profile general_alpha
```

List profiles:

```bash
python -m phragmen.cli profiles
```

---

## JSON schema example

This is an illustrative example (minimal fields shown):

(input quota-floor ballot weights, shares, and quota_floors greater than quota_floor = 1/3 are always reduced to 1/3 in the algorithm)

```json
{
  "seats": 10,
  "total_population": 5300000,
  "total_enrollment": 4100000,
  "total_turnout": 3300000,

  "candidates": ["A", "B", "C"],

  "candidate_meta": {
    "A": {
      "megas": ["M02"]
    },
    "D": {
      "megas": ["M01"],
    }
  },

  "ballots": [
    {"weight": 1, "approvals": ["A", "B"], "electorate": "E01"},
    {"weight": 2, "approvals": ["B","E"], "electorate": "E02"}
  ],

  "electorate_ballots": [
    {"id": "E01", "weight": 100000, "turnout": 70000, "candidates": ["A", "B"]},
    {"id": "E02", "weight": 120000, "turnout": 80000, "candidates": ["B", "C"]}
  ],

  "party_ballots": [
    {"id": "P01", "weight": 500000, "candidates": ["A", "B"]}
  ],

  "mega_ballots": [
    {"id": "M01", "weight": 2000000, "candidates": ["A", "B", "C"]},
    {"id": "M02", "share": 0.6},
    {"id": "M03", "quota_floor": "1/2"}
  ],

  "partyrock_ballots": [
    {"weight": 1000, "approvals": ["C","D"], "electorate": "E02", "party": "P01"}
  ],

  "megarock_ballots": [
    {"weight": 200, "approvals": ["A"], "electorate": "E01", "mega": "M01"}
  ]
}
```

MD
