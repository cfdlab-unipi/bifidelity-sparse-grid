# Real Leja node data

`leja.csv` contains 31 classical and symmetric real Leja prefixes on `[-1, 1]`.
Each row describes one prefix length:

| Column | Contents |
| --- | --- |
| `count` | Number of nodes in the prefix |
| `classical_node` | New classical node added at this length |
| `symmetric_node` | New symmetric node added at this length |
| `classical_weights` | Space-separated weights for the complete classical prefix |
| `symmetric_weights` | Space-separated weights for the complete symmetric prefix |

Nodes are stored once and prefixes reuse earlier rows. Classical nodes start
with `1, -1, 0`; symmetric nodes start with `0, 1, -1`. Odd symmetric prefixes
contain complete mirror pairs. The loader maps nodes to each physical domain.

Node ordering, floating-point values, and uniform-probability prefix quadrature
weights are stored as verified reference data. The weights integrate against
a normalized uniform density and sum to one. PDF-weighted statistics compute
weights for the configured distribution separately.

The 31-node limit applies to a one-dimensional rule, not the total number of
multidimensional sparse-grid points. It is not extended automatically. Adding
rows requires verified nested nodes and complete prefix weights, followed by
numerical regression checks. PDF-weighted Leja generation uses the separate
numerical construction in `nodes.py`.
