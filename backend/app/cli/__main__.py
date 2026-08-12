import sys

from backend.app.cli.main import main

if __name__ == "__main__":
    # sys.exit() is REQUIRED, not decorative. `python -m backend.app.cli` runs
    # THIS file, not main.py's own __main__ block, so discarding main()'s return
    # value made every guard-refusal exit 0: `clear-corpus REFUSED` and
    # `db-init WIPE REFUSED` both reported success to CI, shell `&&` chains and
    # the detached-run harness. Only uncaught exceptions ever failed.
    sys.exit(main())
