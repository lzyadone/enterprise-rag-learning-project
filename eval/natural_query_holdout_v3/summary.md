# Retrieval Union Pool V1

This pool combines 2 retrieval systems and deduplicates candidates per query.
Candidate order is deterministically shuffled before labeling so system rank is hidden.

## Setup

- cases: 32
- pool depth per system: 10
- systems: direct_hybrid, planned_v3_hybrid
- total query/chunk pairs: 320
- candidates per query: 10 to 10 (avg 10)

## Existing Judgments

- inherited: 0
- remaining: 320
- completion: 0.00%

## Unique Contributions

| system | candidates found only by this system |
|---|---:|
| direct_hybrid | 0 |
| planned_v3_hybrid | 0 |

Legacy judged candidates outside every current top-depth run: 0.

The union pool supports fair comparison between candidate generators after remaining pairs are judged.
It still estimates relevance through depth pooling rather than exhaustively judging all collection chunks.
