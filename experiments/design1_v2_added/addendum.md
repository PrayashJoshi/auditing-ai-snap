# Design 1 v2 addendum: three added readers

To be committed before the first added call. `freeze.md` and the frozen run are unchanged.

## What is added and why

After the frozen dry run, and after an 8-input probe that checked each model could be reached and returned a valid `report_flags` call (`probe-reachability.jsonl`), three more readers are added. They are outside reader hypotheses 1 to 3, which stay defined on the three frozen readers. Results for the added readers are reported as additional readers or in an appendix, not as confirmatory tests.

| Reader | Inputs | Repeats | Main calls |
|---|---:|---|---:|
| `glm-5.3-flash` | 264 | 1, 2 | 528 |
| `gemma-4-26b-a4b-it` | 264 | 1, 2 | 528 |
| `qwen3.8-27b` | 264 | 1 | 264 |

## Dry run (each added reader, repeat 1, 32 inputs, 96 calls in all)

- The frozen dry run's 24 inputs (the first 3 of each of the 8 class and state cells, in file order), all with an earned target group.
- Plus 8 inputs with an unearned target group: the first unearned-target input of each of the 8 cells, in file order (33880 at none, clear deposits, ambiguous deposits and lead; 24468 documented; 33192 true zero with rent; 40183 at both stray-credit labels). Every cell has unearned-target inputs, so no substitute rule was needed. The frozen dry run did not include unearned-target inputs.
- Validity is the frozen rule (`reader_output_valid`). Retention keeps the frozen bar: a reader is retained when at least 22 of the 24 frozen-dry-run inputs are valid. The 8 unearned inputs are reported separately (`added-dryrun-validity.csv`) and do not decide retention. A dropped reader is reported as dropped.
- The main added run calls only readers retained in `added-dryrun-validity.csv`. The dry run is logged to `added-dryrun.jsonl` and is not scored.

## Unchanged

- Inputs: `inputs.jsonl`, SHA-256 `9b84d77162ba9e75d1e49e969e2d1d656b6142f54dd816388ac55c54b0737839`.
- Reader instructions `faa22e9fc6f7b58e` and tool `report_flags` with confidence `d7bf54860554aac4`; token cap 8192, temperature 0, concurrency 8, forced tool call, same call function and retry code. The new runner stops if any of these differ.
- Scoring rules of `freeze.md` section 7. No frozen file is edited; the added runner only imports the frozen runner.

## Files

- Runner: `experiments/design1_v2_added/run_added.py`, SHA-256 `d8955e59c46c6b51e8847aa66d4084d6c41962b1bd8420dcbcede1349c5a40f9`.
- Results go to `experiments/design1_v2_added/added-results.jsonl` (job ids tagged `design1v2added`, same record format). They are not mixed into the frozen `results.jsonl`. Run settings are written to `added-dryrun-config.json` and `added-run-config.json` before calls.
- Estimated cost: about 1.1 credits for the main run and 0.1 for the dry run, from the probe's mean cost per call (rough; 8 calls per model).
