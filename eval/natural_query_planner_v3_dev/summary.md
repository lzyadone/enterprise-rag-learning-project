# Retrieval Union Pool V1

This pool combines 3 retrieval systems and deduplicates candidates per query.
Candidate order is deterministically shuffled before labeling so system rank is hidden.

## Setup

- cases: 32
- pool depth per system: 10
- systems: direct_hybrid, planned_v2_hybrid, planned_v3_hybrid
- total query/chunk pairs: 351
- candidates per query: 10 to 12 (avg 10.97)

## Existing Judgments

- inherited: 349
- remaining: 2
- completion: 99.43%

## Unique Contributions

| system | candidates found only by this system |
|---|---:|
| direct_hybrid | 0 |
| planned_v2_hybrid | 29 |
| planned_v3_hybrid | 2 |

Legacy judged candidates outside every current top-depth run: 0.

The union pool supports fair comparison between candidate generators after remaining pairs are judged.
It still estimates relevance through depth pooling rather than exhaustively judging all collection chunks.
