# Retrieval Union Pool V1

This pool combines 1 retrieval systems and deduplicates candidates per query.
Candidate order is deterministically shuffled before labeling so system rank is hidden.

## Setup

- cases: 16
- pool depth per system: 10
- systems: direct_hybrid
- total query/chunk pairs: 279
- candidates per query: 13 to 20 (avg 17.44)

## Existing Judgments

- inherited: 0
- remaining: 279
- completion: 0.00%

## Unique Contributions

| system | candidates found only by this system |
|---|---:|
| direct_hybrid | 119 |
| planned_hybrid | 119 |

Legacy judged candidates outside every current top-depth run: 0.

The union pool supports fair comparison between candidate generators after remaining pairs are judged.
It still estimates relevance through depth pooling rather than exhaustively judging all collection chunks.
