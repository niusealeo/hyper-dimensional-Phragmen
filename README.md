# Sequential Phragmén on Crack — FIFO Time-Priority Edition

This repository implements a generalized, auditable sequential Phragmén engine with:

- **FIFO (time-priority) credit spending** using a cutoff time **τ**
- **soft quota floors** for `electorate`, `party`, and `mega` groups
- quota groups that are **either active (unsatisfied)** or **dormant (satisfied)**  
  - dormant groups **accumulate reserve** but **do not contribute** to selection
- intervention-compatible sequencing (prefix allow-only, iterative allow-only, bans)
- **multi-pass A/B iteration** with **cycle/twin signature detection**
- **franchise (projection) accounting** for electorate inclusion / participation analysis
- auditable CSV outputs for every round

---

## Credit model (FIFO)

Each group *g* earns credit continuously at rate `w_g`.

FIFO mode stores, per group:

- `t_start[g]`: start time of *currently unspent* credit
- `t_now`: global time

Unspent credit at time `t_now`:

```
B_g(t_now) = w_g × max(0, t_now − t_start[g])
```

---

## FIFO spend mode and cutoff τ

When a candidate is elected, spend exactly **1.0 seat value** using **oldest credit first**.

Given paying groups `S`, find a cutoff time `τ ≤ t_now` such that:

```
Σ_g∈S w_g × max(0, τ − t_start[g]) = 1
```

Then advance each paying group’s `t_start[g]` up to `τ` (or to `t_now` if fully drained).

This preserves “overshoot leftovers” automatically: credit earned after `τ` remains intact.

---

## Soft quota floors: active vs dormant

Quota groups (`electorate`, `party`, `mega`) are **minimum quota floors**.

They behave as **reserve racers**:

- **Dormant (satisfied)**: they keep accumulating reserve, but do not affect dt/have.
- **Active (unsatisfied)**: they contribute to dt/have and can be spent.

**Quota activation is computed strictly at the current sequential round `r`**
(no projection look-ahead):

```
required = ceil(quota_floor × r)
active iff winners_in_set < required
```

---

## Tiered priority spending

Spend is performed by priority tiers across kinds.

Default:

```
base > electorate > party > mega
```

You can group kinds in a tier:

```
base > party > electorate,mega
```

Within a tier:

- `combined_fifo` (default): pool kinds, compute one τ
- `separate_by_kind`: FIFO spend kind-by-kind in listed order

---

## Franchise participation (projection)

Projection measures electorate inclusion:

- Only `base` (voter) groups contribute.
- Each base group is counted at most once.
- Per winner:
  - `delta_voter_ballots_used`
  - `delta_projection = delta / total_voter_ballots`
  - `total_projection` cumulative

Projection drives convergence signatures and the “full chamber” rule.

---

## Multi-pass A/B iteration + cycle detection

Pass 1: normal sequential run.

For pass ≥ 2:

- **A**: allow-only winners covering projection interval **[1/9, 5/9]** from previous pass
- **B**: allow-only the entire A list (in order), then run until projection **> 5/9 (strict)**

Signature = “Part-B prefix of winners until projection > 5/9 (strict)”.

All signatures are stored; any repeat signature is a **twin / cycle**.
Cycle length is detected as the difference in iteration indices.

---

## Full chamber completion rule

Full chamber size is:

```
max(
  input seats,
  first round where projection > 2/3 (strict)
)
```

Capped by the number of candidates.

---

## Outputs

Each pass writes:

- `*_rounds.csv` — dt/have/time + projection + quota activation + intervention usage
- `*_quota.csv` — quota group active flags + reserve balances
- `*_projection.csv` — projection accounting per round
- optional `--quota_meta_csv` — normalized population/weight/share/quota_floor

---

## Run

```bash
python -m phragmen.cli election.json --outdir out \
  --profile general_alpha \
  --spend_tiers "base>party>electorate,mega" \
  --tier_within_mode combined_fifo
```
