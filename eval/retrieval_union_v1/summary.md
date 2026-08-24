# Retrieval Union Pool V1

This pool combines five retrieval systems and deduplicates candidates per query.
Candidate order is deterministically shuffled before labeling so system rank is hidden.

## Setup

- cases: 8
- pool depth per system: 10
- systems: direct_dense, direct_bm25, direct_hybrid, planned_dense, planned_hybrid
- total query/chunk pairs: 224
- candidates per query: 23 to 35 (avg 28)

## Existing Judgments

- inherited: 128
- remaining: 96
- completion: 57.14%

## Unique Contributions

| system | candidates found only by this system |
|---|---:|
| direct_dense | 22 |
| direct_bm25 | 26 |
| direct_hybrid | 0 |
| planned_dense | 16 |
| planned_hybrid | 13 |

Legacy judged candidates outside every current top-depth run: 22.

The union pool supports fair comparison between candidate generators after remaining pairs are judged.
It still estimates relevance through depth pooling rather than exhaustively judging all collection chunks.
