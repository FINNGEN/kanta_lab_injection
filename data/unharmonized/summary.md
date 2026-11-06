# Unharmonized suggestions

Input `/mnt/disks/data/kanta/engine/kanta_R15_dev_2026_11_05/kanta_R15_dev_2026_11_05.parquet`. 516 groups (test × unit, rows with a numeric value), 3,204,873 rows: 64 decidable with ≥ 1,000 failed rows (963,706 rows), 452 not decidable with ≥ 1,000.

| Result | Confidence | Groups | Rows |
| --- | --- | ---: | ---: |
| Suggest fix | High | 23 | 799,265 |
| Suggest fix | Medium | 22 | 73,103 |
| Unit is right: approve the Usagi row | High | 3 | 7,686 |
| Unit is right: approve the Usagi row | Medium | 3 | 32,293 |
| Review: values and ranges disagree |  | 4 | 8,067 |
| No unit fits |  | 9 | 43,292 |
| Not decidable: test not in Usagi |  | 175 | 1,075,734 |
| Not decidable: test only unchecked in Usagi |  | 246 | 971,369 |
| Not decidable: too few harmonized rows |  | 31 | 194,064 |

| Concept | Groups | Rows |
| --- | ---: | ---: |
| Concept taken from the test's other units | 15 | 797,154 |
| Mapped in Usagi, conversion failed | 17 | 56,519 |
| No concept in Usagi | 421 | 2,047,103 |
| Pair waiting for Usagi review | 60 | 288,932 |
| Test maps to several concepts | 3 | 15,165 |
