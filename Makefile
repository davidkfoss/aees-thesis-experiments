.PHONY: plots tables artifacts

# Regenerate all figures into reproduced_artifacts/figures/.
plots:
	./scripts/plots/run_all.sh

# Regenerate all tables into reproduced_artifacts/tables/.
tables:
	./scripts/tables/run_all.sh

# Regenerate everything.
artifacts: tables plots
