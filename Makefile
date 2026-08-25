SHELL := /usr/bin/env bash
.RECIPEPREFIX := >

REPO_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= python3
SCRIPTS := transcribe whisperx-benchmark
BASH_SCRIPTS :=
COMMANDS := transcribe whisperx-benchmark
CONFIG_TEMPLATE := $(REPO_DIR)/config.example.toml
CONFIG_LOCAL := $(REPO_DIR)/config.local.toml

PREFIX ?= $(HOME)/.local
BINDIR ?= $(PREFIX)/bin

.PHONY: all install uninstall preflight test help dataset-setup dataset-verify longform benchmark-smoke benchmark-dev-sweep benchmark-test-single benchmark-test-dual result-validate

all:
>@echo "Run 'make preflight' to validate the environment, then 'make install' to install scripts."

preflight: test
>@command -v bash >/dev/null 2>&1 || { echo "ERROR: bash is required" >&2; exit 1; }
>@command -v docker >/dev/null 2>&1 || { echo "ERROR: docker is required for primary workflow" >&2; exit 1; }
>@command -v find >/dev/null 2>&1 || { echo "ERROR: find is required" >&2; exit 1; }
>@for script in $(SCRIPTS); do \
  [ -f "$(REPO_DIR)/$$script" ] || { echo "ERROR: missing $$script" >&2; exit 1; }; \
  [ -x "$(REPO_DIR)/$$script" ] || { echo "ERROR: script not executable: $$script" >&2; exit 1; }; \
done
>@test -f "$(CONFIG_TEMPLATE)" || { echo "ERROR: missing config template: $(CONFIG_TEMPLATE)" >&2; exit 1; }
>@test -f "$(CONFIG_LOCAL)" || { echo "ERROR: copy config.example.toml to config.local.toml and set local paths" >&2; exit 1; }
>@echo "preflight: OK"

install:
>@mkdir -p "$(BINDIR)"
>@ln -sf "$(REPO_DIR)/transcribe" "$(BINDIR)/transcribe"
>@ln -sf "$(REPO_DIR)/whisperx-benchmark" "$(BINDIR)/whisperx-benchmark"
>@for cmd in $(COMMANDS); do \
  echo "installed: $(BINDIR)/$$cmd"; \
done

uninstall:
>@for cmd in $(COMMANDS); do \
  rm -f "$(BINDIR)/$$cmd"; \
  echo "removed: $(BINDIR)/$$cmd"; \
done

test:
>@command -v bash >/dev/null 2>&1 || { echo "ERROR: bash is required" >&2; exit 1; }
>@command -v $(PYTHON) >/dev/null 2>&1 || { echo "ERROR: $(PYTHON) is required" >&2; exit 1; }
>@for script in $(SCRIPTS); do \
  [ -f "$(REPO_DIR)/$$script" ] || { echo "ERROR: missing $$script" >&2; exit 1; }; \
  [ -x "$(REPO_DIR)/$$script" ] || { echo "ERROR: script not executable: $$script" >&2; exit 1; }; \
  echo "found: $(REPO_DIR)/$$script"; \
done
>@if [ -n "$(BASH_SCRIPTS)" ]; then \
  for script in $(BASH_SCRIPTS); do \
    bash -n "$(REPO_DIR)/$$script"; \
    echo "test: checked $$script"; \
  done; \
fi
>@PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover \
  -s "$(REPO_DIR)/tests" -p 'test_*.py' -v
>@echo "test: OK"

dataset-setup:
>$(PYTHON) scripts/librispeech_corpus.py download --root /data/datasets/LibriSpeech

dataset-verify:
>$(PYTHON) scripts/librispeech_corpus.py verify --root /data/datasets/LibriSpeech

longform:
>@echo "Use: python3 scripts/build_librispeech_longform.py --source /data/datasets/LibriSpeech/dev-clean --destination /data/datasets/LibriSpeech-longform/dev-clean"

benchmark-smoke:
>./whisperx-benchmark --config config.local.toml --dataset /data/datasets/LibriSpeech/dev-clean --ext flac --max-files 8 --output-root benchmark-results/smoke --set cuda-devices=0 --set beam-size=1 --set no-align=true --trace

benchmark-dev-sweep:
>./whisperx-benchmark --config config.local.toml --dataset /data/datasets/LibriSpeech/dev-clean --ext flac --output-root benchmark-results/dev-sweep --sweep batch-size=8,16,24,32 --set beam-size=1 --set no-align=true --trace

benchmark-test-single:
>@echo "Locked configuration required. Example: python3 scripts/repeated_benchmark.py --config config.local.toml --dataset /data/datasets/LibriSpeech/test-clean --output-root benchmark-results/test-clean-locked --batch-size 16 --beam-size 1 --alignment disabled"

benchmark-test-dual:
>@echo "Locked configuration required. Example: python3 scripts/dual_gpu_benchmark.py --config config.local.toml --dataset /data/datasets/LibriSpeech/test-clean --output-root benchmark-results/test-clean-dual --batch-size 16 --beam-size 1"

result-validate:
>@test -n "$(RESULT)" || { echo "usage: make result-validate RESULT=benchmark-results/.../run_001" >&2; exit 2; }
>$(PYTHON) scripts/validate_benchmark_result.py "$(RESULT)"

help:
>@echo "make preflight"
>@echo "  Validate local prerequisites for running scripts."
>@echo "make install [BINDIR=path|PREFIX=path]"
>@echo "  Install transcribe and whisperx-benchmark into $(BINDIR)"
>@echo "make uninstall [BINDIR=path|PREFIX=path]"
>@echo "  Remove transcribe and whisperx-benchmark from $(BINDIR)."
>@echo "make test"
>@echo "  Run offline control-plane behavior tests."
