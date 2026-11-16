# Data

Raw files are not committed (`data/raw/`, git-ignored). Run `bash data/fetch.sh` or symlink existing copies. Checksums: `SHA256SUMS`.

Source: SNAP Quality Control public-use database, https://snapqcdata.net/datafiles. Prepared by Mathematica for USDA Food and Nutrition Administration (FNA; older documents say FNS). No suggested citation exists; the constructed citation used here:

> U.S. Department of Agriculture, Food and Nutrition Administration. (2026). *SNAP Quality Control Database, Fiscal Year 2024, public-use files*. Prepared by Mathematica. Revised version posted 18 August 2026. https://snapqcdata.net/datafiles

Technical documentation: Leftin, J., Monkovic, M., Yang, F., Bonelli, S., & Maula, J. C. (2026). *Technical documentation for the fiscal year 2024 SNAP Quality Control database and the QC Minimodel* (Final report). Mathematica. No terms-of-use or licence statement was found on the pages read there.

| File (data/raw/) | Fiscal year | Source URL | Notes |
|---|---|---|---|
| fy2024/qc_pub_fy2024.csv (from qcfy2024_csv.zip) | FY2024 | https://snapqcdata.net/sites/default/files/2026-08/qcfy2024_csv.zip | 44,891 rows x 1,177 cols (row count reproduced in `results/01-data-checks`). Zip posted under 2026-08 (revised weights); the CSV inside is timestamped 2026-06-26 and carries the corrected HWGT/FYWGT (checked in `results/01-data-checks`, section 2). |
| fy2023/qc_pub_fy2023.csv (from qcfy2023_csv.zip) | FY2023 | https://snapqcdata.net/sites/default/files/2025-03/qcfy2023_csv.zip | 43,776 rows x 854 cols (published figure; not re-checked here). |
| fy2022/qc_pub_fy2022.csv (from qcfy2022_csv.zip) | FY2022 | https://snapqcdata.net/sites/default/files/2024-05/qcfy2022_csv.zip | 41,391 rows x 816 cols (published figure; not re-checked here). |
| forms/michigan/*.pdf, *.txt | n/a | michigan.gov (MDHHS-1171 application, Rev. 10-26; information booklet Rev. 10-22) | Official state form; downloaded manually, not scripted. Texas H1010 and California CF 285 were not used. |

Only the FY2024 counts are re-checked in this repo.

`processed/hr1_state_snap_exposure.csv` is a state-level table of payment error rates and H.R. 1 cost-share exposure, written by `supplementals/hr1-policy-stakes.py` from the public policy documents it lists under `data/raw/policy/` (not committed).
