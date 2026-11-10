# Unharmonized suggestions

Input `/mnt/disks/data/kanta/engine/kanta_R15_dev_2026_11_06/kanta_R15_dev_2026_11_06.parquet`. 514 groups (test × unit, rows with a numeric value), 2,531,898 rows: 61 decidable with ≥ 1,000 failed rows (267,476 rows), 453 not decidable with ≥ 1,000.

| Result | Confidence | Groups | Rows |
| --- | --- | ---: | ---: |
| Suggest fix | High | 21 | 126,290 |
| Suggest fix | Medium | 22 | 73,103 |
| Unit is right: approve the Usagi row | High | 3 | 7,686 |
| Unit is right: approve the Usagi row | Medium | 2 | 9,038 |
| Review: values and ranges disagree |  | 4 | 8,067 |
| No unit fits |  | 9 | 43,292 |
| Not decidable: test not in Usagi |  | 175 | 1,075,734 |
| Not decidable: test only unchecked in Usagi |  | 246 | 971,369 |
| Not decidable: too few harmonized rows |  | 32 | 217,319 |

| Concept | Groups | Rows |
| --- | ---: | ---: |
| Concept taken from the test's other units | 14 | 125,946 |
| Mapped in Usagi, conversion failed | 17 | 56,519 |
| No concept in Usagi | 421 | 2,047,103 |
| Pair waiting for Usagi review | 59 | 287,165 |
| Test maps to several concepts | 3 | 15,165 |

## Secondary-injection rules

| Validation | Rules |
| --- | ---: |
| Confirmed | 47 |
| No effect | 2 |
| Not confirmed | 2 |
| Cannot check | 26 |
| Not applied | 2 |
