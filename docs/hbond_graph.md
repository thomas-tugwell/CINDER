# Hydrogen-bond graph (first and second shell)

`cinder.objectives.hbond_graph` (backend-independent) and `cinder.energy.tmol_roles` (reads roles
from a TMol pose). Enabled in the design energy with `energy.hbond_graph_weight > 0`.

## Why it exists

`interface = E(complex) - E(target) - E(binder)` removes every intra-chain contribution by
construction. A residue that never touches the target but hydrogen-bonds to, and positions, a
residue that does is therefore invisible to it. The graph counts those *second-shell* bonds, and
cooperativity between shells, with differentiable quantities.

## Definitions

* **Nodes:** atoms that can donate or accept a hydrogen bond. Donor/acceptor atom types and the
  polar hydrogens come from TMol's own hydrogen-bond database (no residue-name tables). Only atoms
  of residues within 8 Å of the other chain (a fixed region, including second-shell residues) are
  nodes.
* **Edge strength** for donor heavy atom `D`, hydrogen `H`, acceptor `A`:

      a = sigmoid((r_cut - |H-A|) / w_r) * sigmoid((-cos(D-H...A) - cos_cut) / w_a)

  (H...A close, and D-H...A close to linear; defaults `r_cut = 2.5 Å`, angle > 120°). Pairs in the
  same residue, adjacent residues of one chain, or the same atom are masked.
* **First shell** (a node bonds across the interface): `n1_v = 1 - exp(-sum_u S_vu X_vu)`, with
  `X` the cross-chain mask.
* **Second shell** (a node bonds, within its own chain, to a first-shell node):
  `n2_v = (1 - n1_v)(1 - exp(-sum_u S_vu (1 - X_vu) n1_u))`.

Shells are soft (propagated through the adjacency), so there are no hard cutoffs.

## Objectives (`HBondGraphConfig` weights; the energy is the negative score)

| Quantity | Meaning | Default weight |
|---|---|---|
| first-shell strength | `sum a_da X_da`, bonds across the interface | 1.0 |
| second-shell support | `sum a_da (1-X_da) s_da`, `s = 1-(1-n1_d)(1-n1_a)` | 0.5 |
| cooperativity | `sum_v c_v i_v / (c_v + i_v)`, smooth min of cross-chain and intra-chain bonding at a node | 0.25 |
| buried unsatisfied | `sum_v burial_v exp(-deg_v / tau)`, penalty | 0.5 |

Burial is the soft count of heavy atoms within 6 Å of a polar atom (midpoint 45, which is near the
75th percentile in barnase/barstar; polar atoms there have 25 to 51 neighbours at the 10th to 90th
percentile). These are heuristics and are configurable (`energy.options.hbond_graph`).

## Independent check

`hard_hbond_report` counts bonds with hard thresholds on the heavy-atom `D...A` distance (<= 3.5 Å)
and the `D-H...A` angle (>= 110°), plus first/second-shell atoms and buried unsatisfied polars.
Never use the optimised graph score as its own validation; use this counter (or an external tool).
On barnase/barstar it finds 15 cross-interface bonds (the soft first-shell strength is 13.2) and
17 buried unsatisfied polar atoms.

## Design positions from the shells

`propose_design_positions(system, shells="first"|"second"|"both")` lists the binder positions in
the native pose's shells. On barnase/barstar the first shell recovers the classic barstar
interface residues (Y29, G31, N33, L34, D35, D39, T42, G43, E76) from geometry alone; the
binder-side second shell is empty for this complex. Treat the output as a list to inspect and edit.

## Cautions

* A graph reward can be gamed with unrealistic geometry: combine it with `repulsion`, `strain`
  and, for flexible binders, `internal`.
* Whether stronger second-shell networks improve binding is a hypothesis, not an established
  result; nothing here has been compared with experiment.
