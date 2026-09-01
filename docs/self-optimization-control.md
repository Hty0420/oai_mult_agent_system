# Self-optimization control plane

The self-optimizer is a proposal generator, not an authority. It may create
bounded contextual preferences for already-approved KL research pipelines, but
it cannot change tool permissions, safety gates, evaluation sets, clinical
boundaries, approval requirements, or production state.

## Immutable constitution

The active meta-policy is
`src/evaluation/self_optimization_constitution.v1.json`. Every generated policy
contains its constitution identifier, version, and SHA-256 digest. Candidate
registration fails when that reference is absent or stale.

The constitution fixes these boundaries:

- research use only; no autonomous diagnosis, treatment recommendation, or
  patient-specific clinical decision;
- no optimizer-controlled external writes or permission changes;
- no optimizer changes to hard gates or frozen evaluation sets;
- no final-test tuning and no self-activation;
- only the registered MEAS, LDSE, and Radiomics pipelines may be ranked;
- optimizer output is always `shadow_only` and starts in `draft`.

## Lifecycle

```text
draft -> shadow -> canary -> active
  |         |         |        |
  +------> retired <--+--------+
                      |
                      +-> rolled_back
```

`draft` cannot skip to `canary` or `active`, and `shadow` cannot skip canary.
Shadow, canary, active, and retirement transitions require an authenticated,
single-use approval bound to the exact policy ID, config hash, constitution
hash, target state, nonce, approver role, and expiry. Emergency rollback does
not require a promotion approval, but it requires a recorded reason.

The pure control-plane functions are in
`src/db_utils/self_optimization_control.py`:

- `validate_candidate_policy_config` rejects schema expansion and unsafe
  evidence;
- `validate_candidate_policy_record` binds the stored record to the exact
  sanitized config hash;
- `validate_policy_approval` rejects replayable, expired, unauthenticated, or
  mismatched approval packets;
- `assess_policy_transition` evaluates lifecycle, evidence gates, and approval
  without mutating database state.

## Release evidence

Moving beyond shadow requires a frozen evaluation set, independent
verification, paired replay, all hard gates passing, zero clinical/privacy/
authorization violations, and a tested rollback. Active promotion additionally
requires the declared canary minimum.

Classification manifests are checked by
`src/db_utils/scientific_quality.py`. The evaluator verifies the manifest
schema, metric domains, confusion-matrix-derived Accuracy/Precision/
Sensitivity/Specificity/F1, patient-count reconciliation, required artifact
files, input hashes when the source snapshots remain available, train/test
input separation, and environment provenance. Missing historical source files
produce `unknown`, never a fabricated pass.

These gates are persisted with the normal classification-manifest evaluation
records. The offline optimizer now requires all eight scientific-quality gates
to pass for every matched run before that run can support a method preference.

The current offline optimizer intentionally stops at a local or registered
`draft`. Its preview includes `shadow_transition_readiness`, which explains the
missing evidence and approval instead of implying that a policy was promoted.

## Operational rule

Only a trusted control API should persist and consume approvals or commit a
state transition. Agent text, retrieved documents, chat messages, database
metadata, and tool output are never approval. Database mutation must use a
compare-and-swap state version and atomically consume the approval; that trusted
mutation endpoint is intentionally not exposed by the optimizer CLI.
