# Code Standards

## Style
- Follow **PEP 8** (4-space indent, snake_case for variables/functions, descriptive names)
- Every function gets a short docstring: what it does, its parameters, what it returns
- Keep functions focused — one clear responsibility each; prefer several small functions over one long one

## Project-Specific Conventions
- Data mining logic lives in `src/`, not inline in `app.py` — `app.py` should mostly call functions and render results
- No magic numbers — thresholds (e.g., risk tier cutoffs) go in named constants at the top of the relevant file
- Every model training script prints its evaluation metrics before saving the model — never save a model silently

## While Still Learning Python
- Prefer clarity over cleverness — a slightly longer but readable block beats a dense one-liner
- Comment *why*, not *what*, when a step isn't obvious (e.g., why a feature was dropped)
- It's fine to leave a `# TODO` comment for something to revisit rather than over-engineering it on the first pass

## Git
- Commit in small, working increments (e.g., "add data cleaning", "train baseline approval model") rather than one large commit
- Don't commit `data/raw/` if the dataset is large — note the Kaggle source in the README instead
