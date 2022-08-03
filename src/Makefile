.PHONY: help train results

PY = python3

# Training
EPOCHS ?= 10
BATCH_SIZE ?= 32

# Results
SESSION ?= -1
EPOCH ?= -1

help:
	@echo "Help:"
	@echo "- make help: Show this message."
	@echo "- make train: Train the model."
	@echo "- make results: Show results."

train:
	$(PY) train.py --epochs $(EPOCHS) --batch-size $(BATCH_SIZE)

results:
	$(PY) results.py --session $(SESSION) --epoch $(EPOCH)
