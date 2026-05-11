# LOIS

This repository contains the implementation for the paper *Locally Optimal Solutions for Integer Programming Games*.

Files:

- `cng.py`: synthetic critical node game generator plus payoff and feasibility helpers.
- `lois1.py`: LOIS-1 mixed-integer model.
- `bilevel_lois1.py`: defender-leader bilevel model.
- `summarize_results.py`: small CSV summarizer for paper tables.

## Dependencies

Install:

```bash
pip install -r requirements.txt
```

## Usage

Run the LOIS-1 model on one synthetic instance:

```bash
python lois1.py --size 50 --seed 6318
```

Run the bilevel variant:

```bash
python bilevel_lois1.py --size 50 --seed 6318
```

Summarize a batch of CSV outputs:

```bash
python summarize_results.py --prefix lois1cng --columns t:mean PoD:mean PoA:mean
python summarize_results.py --prefix bilevel-lois1 --columns t:mean PoS:mean
```

## Notes

- `cng.py` includes a self-contained knapsack helper used for payoff normalization.
- Synthetic instances are generated directly from the model parameters used in the paper.
- To reproduce paper-scale tables, run many seeds and save each script's stdout into CSVs for aggregation.
