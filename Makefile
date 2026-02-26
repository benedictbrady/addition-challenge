SUBMISSION ?= starter/submission.py

install:
	uv sync

validate:
	uv run addition validate $(SUBMISSION)

eval:
	uv run addition eval $(SUBMISSION)

eval-quick:
	uv run addition eval $(SUBMISSION) --num-random 100

count:
	uv run addition count-params $(SUBMISSION)

train:
	uv run python starter/train.py

test:
	uv run pytest tests/ -v
