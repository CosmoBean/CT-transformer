# PDF Report Samples

These PDF files are rendered from the existing Claude review bundles and are
intended for quick manual review.

- `50a418190bc3fb1ef1633bf9678929b3.pdf`
  - normal case
- `9a5094b2563a1ef3ff50dc5c7ff71345.pdf`
  - cleaner abnormal case with localization support
- `e9954e6e3b2d0c5bf990a519c0ba5abe.pdf`
  - complex multi-label case requiring review

They were generated with:

```bash
python scripts/export_review_pdfs.py \
  --inputs \
    experiments/claude_review/tightened_showcase/50a418190bc3fb1ef1633bf9678929b3 \
    experiments/claude_review/demo_smoke/9a5094b2563a1ef3ff50dc5c7ff71345 \
    experiments/claude_review/tightened_showcase/e9954e6e3b2d0c5bf990a519c0ba5abe \
  --output-dir reports/pdf_samples
```
