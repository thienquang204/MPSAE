# Implementation verification

Checked on 2026-11-25 with Python 3.13 and CPU PyTorch 2.9.1.

- All 17 new paper-reproduction tests passed: catalog counts, original receipt
  import, duplicate/conflict handling, complete training identities, sparsity
  checks, report exports, and evaluation without SAE optimizer steps.
- The combined targeted suite reported **87 passed, 3 failed**. The three
  failures also reproduce on the unchanged `e504eb7` checkout. They are existing
  `test_campaign_runner.py` downstream test-double failures involving duplicate
  `exclude_self` arguments; they are not fixed by this organization change.
- Existing retrieval smoke/resume, patch training/sweeps/early stopping, and
  79-evaluation planning tests passed in that targeted suite.
- All 13 LaTeX table templates compiled into a preview PDF. Template checks use
  explicitly missing scores; synthetic test scores are never paper evidence.
- Windows verification used extended-length temporary paths to avoid the
  filesystem limit encountered by deeply nested existing cache directories.

No production dataset was downloaded and no full GPU experiment was run during
this change. Original run receipts are still needed to validate the historical
paper numbers, encoder revisions, and actual convergence records.

On the intended Linux server, run the new tests with:

```bash
python -m pytest tests/test_paper_reproduction.py -q
```
