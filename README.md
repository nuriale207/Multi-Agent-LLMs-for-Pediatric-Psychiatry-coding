# Multi-agent LLMs and fine-tuned transformers for pediatric psychiatry coding

Code to predict psychiatric ICD-10 ("F" chapter) diagnostic codes from
free-text clinical notes, accompanying the paper:

> _Add full citation here once available (authors, title, venue, year, DOI)._

```bibtex
@article{TODO,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO},
  doi     = {TODO}
}
```

This repository contains **only the code**: it does not include the trained
model weights or the clinical data used in the paper, both of which are
private — see the data availability notes in each subproject's README. The
paper compares two independent approaches to the same task, each in its own
self-contained directory:

- **[`transformer_classification/`](transformer_classification/README.md)**
  — fine-tuned transformer classifiers (BERT / Longformer): train, test,
  re-binarize, and evaluate (micro/macro F1, patient aggregation,
  demographic breakdown, note-voting).
- **[`llm_prompting/`](llm_prompting/README.md)** — a multi-agent prompting
  pipeline built on **Qwen2.5-7B-Instruct**: an extraction agent pulls
  diagnoses out of each note, retrieval matches them to ICD-10 codes, and
  further agents review and reconcile the patient-level code list.

Each directory has its own README (install, expected data format, usage,
data availability) and its own `requirements*.txt` — the two pipelines need
incompatible dependency versions, so they are meant to run in separate
environments.

## License

Released under the [MIT License](LICENSE).
