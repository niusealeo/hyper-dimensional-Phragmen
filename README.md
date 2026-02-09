# Sequential Phragmén with Projection, Quota Floors, and Iterative Convergence
"Modifiable iterative hyper-dimensional seq Phragmén"

This repository implements an **extended Sequential Phragmén election engine** designed for large, heterogeneous electorates with:

- weighted voter ballots  
- quota-floor “mega ballots” and party ballots  
- explicit franchise-participation measurement (“projection”)  
- deterministic iterative convergence under controlled interventions  

It is intended for **research, simulation, and institutional design**, not as a drop-in replacement for statutory election software.

---

## 1. What this is (and is not)

### This **is**
- A **credit-based Sequential Phragmén** implementation
- With **soft quota floors** (not hard constraints)
- Supporting **identity-based and organisational voting blocs**
- Measuring **actual voter participation** independently from quotas
- Capable of **iterative re-selection** with convergence testing
- Able to **re-fill vacancies** in an already-elected chamber

### This is **not**
- A simple approval count
- A divisor method (D’Hondt, Sainte-Laguë, etc.)
- A static optimisation (no global solve)
- A hard-quota or reserved-seat system

---

## 2. Core concepts

### 2.1 Base voter ballots
- Ordinary approval ballots
- Can be weighted
- Identical approval sets are **canonicalised into groups** for efficiency
- These ballots define **franchise participation**

### 2.2 Mega ballots
Mega ballots represent **identity groups, organisations, or constituencies**.

They:
- Earn credit over time (like voters)
- Activate only when their **quota floor** is threatened
- Accumulate reserve while inactive
- Do **not** count toward franchise participation

Mega ballots can be defined in **two equivalent ways**.

#### Mode A — membership based
```json
{
  "id": "union_X",
  "abs_weight": 1200,
  "population": 59899,
  "approvals": ["A", "B", "C"]
}
```
#### Mode B — membership based
```json
{
  "id": "union_X",
  "quota_floor": 0.25,
  "auto_include": { "groups_any": ["union"] }
}
```

Quota-floor inversion rules:
- If ```quota_floor < 1/3``` → ```share = (3/2) * quota_floor```
- If ```quota_floor == 1/3``` → ```share = 1/2``` (minimal saturation assumption)

Relative voting weight is always normalised against **total voter ballots**.

---

### 2.3 Party ballots

Party ballots are a special case of mega ballots:
- They behave **exactly like mega ballots** in the credit race
- Their candidate list is **not a ranking**
- List order is used only to break ties between equal-time candidates

---

## 3. Projection: measuring franchise participation

This implementation introduces a projection fraction ```p```, which measures:
> *What proportion of the electorate has actually participated in electing at least one seat?*

### Definition

- Only **base voter ballots** count
- Each voter ballot can contribute at **most once**
- Mega/party ballots do **not** contribute
- For each round:
    - Count newly-used voter ballots
    - Divide by total voter ballots
    - Accumulate into ```p_total```

A seat elected mostly by mega/party credit may advance time but add **little or no projection**.

This decouples:
- **Representation power** (quotas, blocs)
- From **participation coverage** (voters)

---

## 4. Iterative convergence process

The engine supports **controlled iterative reselection**.

### Pass 1
- Normal sequential run
- Record projection intervals ```(p_prev, p_curr]``` for each winner

### Iteration i ≥ 2

Each iteration has two phases:

#### Part A
- Allow-only pool = winners covering projection range **[1/9, 5/9]**
- Run until projection **strictly exceeds 5/9**

#### Part B
- Allow-only pool = entire Part-A winner list
- Run until projection **strictly exceeds 5/9**

#### Convergence test
- Compare Part-B prefix (up to projection > 5/9)
- If identical to previous iteration → **converged**

### Limits
- Default: 19 iterations
- If not converged, the CLI **prompts the user** to continue
- Batch mode can disable prompting

---

## 5. Full chamber completion (after convergence)

Once convergence is reached, the election is **completed from scratch**.

The final chamber size is:
```sql
max(
  input_seats,
  first round where projection > 2/3
)
```

This ensures:
- At least the intended number of seats
- And sufficient **franchise coverage**

---

## 6. Interventions (resignations, exclusions)

The input JSON can specify prefix interventions:
```json
"prefix_intervention": {
  "allow_only": ["Alice", "Bob"],
  "ban": ["Charlie"]
}
```

Rules:
- ```allow_only``` candidates are **consumed first**, in whatever order the algorithm selects
- After the list is exhausted, normal selection resumes
- ```ban``` candidates are never eligible

This is designed for:
- Filling vacancies
- Rerunning elections with a fixed chamber minus some members
- Institutional continuity scenarios

Prefix interventions always take priority over iterative pools.

---

## 7. Input JSON structure (overview)
```json
{
  "seats": 120,
  "candidates": ["A","B","C"],

  "ballots": [
    {"approvals":["A","B"], "weight":1}
  ],

  "candidate_meta": {
    "A": {"groups":["urban"], "tags":["youth"]}
  },

  "mega_ballots": [...],
  "party_ballots": [...],

  "prefix_intervention": {...}
}
```

All sections are optional except ***seats***.

8. Output files

For each run / pass, CSVs are written to ```--outdir```:

```*_rounds.csv```

Per-round detail:
- winner
- time increment
- credit used
- projection delta and cumulative projection
- quota activation status
- intervention usage

```*_projection.csv```

Compact participation curve:
- round
- delta ballots used
- cumulative projection

```*_quota.csv```

Long-form quota diagnostics:
- per quota group, per round
- activation, reserve balance, required counts

Optional
- quota_meta_csv: normalised mega/party definitions

---

## 9. Running the script

Interactive (default):

```bash
python phragmen_seq.py election.json --outdir out
```

Batch mode:

```bash
python phragmen_seq.py election.json --no_prompt --max_iters 19
```
---

## 10. Design notes
- All quota rules are **soft racing conditions**
- No candidate is guaranteed a seat
- No seat is “reserved”
- Everything is auditable via CSV
- Memory usage scales with **approval groups**, not raw ballots

---

## 11. Status

This code version is:
- Research-grade
- Deterministic
- Suitable for large-scale simulation, comparative analysis, and independent result verification
- Not certified for statutory elections

---
