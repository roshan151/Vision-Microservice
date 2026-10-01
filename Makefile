# Vision service: install, test and run it on the Mac mini.
#
#   make help      list every target
#   make deploy    after any code change: install deps if needed, test, restart
#
# Everything here is plain launchd + a Python venv. See README.md, "Deploying".

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Optional local settings (VS_PORT, VS_DEVICE, ...), shared with run.sh.
-include .env
VS_PORT ?= 8080

PYTHON         ?= python3
VENV           := .venv
PY             := $(VENV)/bin/python
STAMP          := $(VENV)/.installed

LABEL          := com.roshan.visionservice
DOMAIN         := gui/$(shell id -u)
PLIST_SRC      := deploy/$(LABEL).plist.in
PLIST_DST      := $(HOME)/Library/LaunchAgents/$(LABEL).plist
LOG_DIR        := $(HOME)/Library/Logs/vision-service
LOG_FILE       := $(LOG_DIR)/vision-service.log

HEALTH_URL     := http://127.0.0.1:$(VS_PORT)/healthz
# First start after install downloads model weights; give it time.
HEALTH_TIMEOUT ?= 180

.PHONY: help install reinstall test run deploy start restart stop uninstall \
        status health logs wait-healthy clean

help: ## Show this list
	@grep -E '^[a-z-]+:.*## ' $(firstword $(MAKEFILE_LIST)) | \
	  awk 'BEGIN{FS=":.*## "}{printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- build ---

install: $(STAMP) ## Create .venv and install deps (only redoes work when pyproject.toml changes)

$(STAMP): pyproject.toml
	@test -d $(VENV) || { \
	  $(PYTHON) -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
	    || { echo "Need Python >= 3.10 (macOS's own python3 is 3.9). Try: brew install python, or make PYTHON=python3.12 install"; exit 1; }; \
	  $(PYTHON) -m venv $(VENV); }
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install -e '.[models,dev]'
	@touch $@

reinstall: ## Delete .venv and install everything from scratch
	rm -rf $(VENV)
	@$(MAKE) --no-print-directory install

test: $(STAMP) ## Run the test suite (~1 s, no models or GPU needed)
	$(PY) -m pytest -q

# -------------------------------------------------------------- service ---

deploy: install test start ## Rebuild after a change: install if needed, test, restart, wait until healthy

# The installed plist is generated from the template, so its paths always
# point at this checkout. Re-rendered whenever the template changes.
$(PLIST_DST): $(PLIST_SRC)
	@mkdir -p $(dir $@) $(LOG_DIR)
	sed -e 's|__REPO_DIR__|$(CURDIR)|g' -e 's|__LOG_FILE__|$(LOG_FILE)|g' $< > $@
	@plutil -lint -s $@

# Unload the job if it is loaded, and wait until launchd has let go of it
# (bootstrapping too soon after a bootout fails with "Input/output error").
define unload
	@if launchctl print $(DOMAIN)/$(LABEL) >/dev/null 2>&1; then \
	  launchctl bootout $(DOMAIN)/$(LABEL) 2>/dev/null || true; \
	  for i in $$(seq 1 50); do \
	    launchctl print $(DOMAIN)/$(LABEL) >/dev/null 2>&1 || break; sleep 0.2; \
	  done; \
	fi
endef

# Refuse to start if something else is listening on the port, and say what it
# is. Otherwise the service loads every model, fails to bind, and launchd
# restarts it every 10 s, forever. Waits a few seconds first, in case it's our
# own previous instance still shutting down.
define port_free
	@for i in $$(seq 1 20); do \
	  lsof -nP -iTCP:$(VS_PORT) -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 0.5; \
	done; \
	if lsof -nP -iTCP:$(VS_PORT) -sTCP:LISTEN >/dev/null 2>&1; then \
	  echo "Port $(VS_PORT) is already in use by:"; \
	  lsof -nP -iTCP:$(VS_PORT) -sTCP:LISTEN; \
	  echo; \
	  echo "Stop that process (kill <PID>, or quit the app it belongs to), or pick"; \
	  echo "another port with VS_PORT=... in .env. Then run this again."; \
	  exit 1; \
	elif nc -z 127.0.0.1 $(VS_PORT) >/dev/null 2>&1; then \
	  echo "Port $(VS_PORT) is in use by a process owned by another user."; \
	  echo "See which with: sudo lsof -nP -iTCP:$(VS_PORT) -sTCP:LISTEN"; \
	  exit 1; \
	fi
endef

start: $(STAMP) $(PLIST_DST) ## (Re)start the service in the background; it also starts at every login
	$(unload)
	$(port_free)
	@mkdir -p $(LOG_DIR)
	@# Keep one previous log; otherwise access logs at 2 Hz grow without bound.
	@if [[ -f $(LOG_FILE) ]]; then mv -f $(LOG_FILE) $(LOG_FILE).1; fi
	launchctl bootstrap $(DOMAIN) $(PLIST_DST)
	@$(MAKE) --no-print-directory wait-healthy

restart: start ## Same as start

stop: ## Stop the service (stays stopped until `make start` or your next login)
	$(unload)
	@echo "stopped"

uninstall: stop ## Stop the service and remove it from launchd for good
	rm -f $(PLIST_DST)

run: $(STAMP) ## Run in this terminal instead of in the background (run `make stop` first)
	@if launchctl print $(DOMAIN)/$(LABEL) >/dev/null 2>&1; then \
	  echo "The background service is running and holds port $(VS_PORT). Run 'make stop' first."; exit 1; \
	fi
	$(port_free)
	./run.sh

# ----------------------------------------------------------- inspection ---

status: ## Is it loaded, running and healthy?
	@launchctl print $(DOMAIN)/$(LABEL) 2>/dev/null \
	  | grep -E '^\s*(state|pid|runs|last exit code) =' \
	  || echo "not loaded (run: make start)"
	@$(MAKE) --no-print-directory health || true

health: ## Print /healthz
	@curl -fsS -m 3 $(HEALTH_URL) | $(PYTHON) -m json.tool \
	  || { echo "no answer from $(HEALTH_URL)"; exit 1; }

logs: ## Follow the service log (Ctrl-C to stop)
	tail -n 100 -F $(LOG_FILE)

wait-healthy:
	@printf "waiting for $(HEALTH_URL) "; \
	for i in $$(seq 1 $(HEALTH_TIMEOUT)); do \
	  if curl -fsS -m 2 $(HEALTH_URL) >/dev/null 2>&1; then \
	    echo " up after $${i}s"; $(MAKE) --no-print-directory health; exit 0; \
	  fi; \
	  if grep -q "address already in use" $(LOG_FILE) 2>/dev/null; then \
	    echo; echo "Port $(VS_PORT) was taken while the service was starting. Stopping it so"; \
	    echo "launchd doesn't keep retrying. Current holder:"; \
	    lsof -nP -iTCP:$(VS_PORT) -sTCP:LISTEN; \
	    launchctl bootout $(DOMAIN)/$(LABEL) 2>/dev/null; exit 1; \
	  fi; \
	  printf "."; sleep 1; \
	done; \
	echo; echo "not healthy after $(HEALTH_TIMEOUT)s. Last lines of $(LOG_FILE):"; \
	tail -n 40 $(LOG_FILE); exit 1

clean: ## Remove caches (keeps .venv)
	find . -name __pycache__ -type d -prune -not -path './$(VENV)/*' -exec rm -rf {} +
	rm -rf .pytest_cache *.egg-info
