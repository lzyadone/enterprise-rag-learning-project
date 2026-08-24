# Natural Query Curation

- raw questions: 38
- curated questions: 32
- removed same-route duplicates: 6
- route counts: {'direct': 18, 'planned': 14}
- bge-m3 cosine threshold: 0.86

| removed | kept | route | cosine |
|---|---|---|---:|
| natural_dev_015 | natural_dev_012 | direct | 0.8672 |
| natural_dev_017 | natural_dev_016 | direct | 0.8666 |
| natural_dev_034 | natural_dev_026 | planned | 0.8790 |
| natural_dev_035 | natural_dev_024 | planned | 0.9241 |
| natural_dev_036 | natural_dev_029 | planned | 0.8789 |
| natural_dev_038 | natural_dev_037 | planned | 0.8875 |

## Boundary

Only later questions with a same-route cosine similarity at or above the threshold were removed. Similar cross-route pairs are retained as useful routing contrasts.
