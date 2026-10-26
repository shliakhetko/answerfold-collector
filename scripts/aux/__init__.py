"""CLI entry points for the post-window auxiliary collection batch.

One module per task, each with `parse_args(argv)` + `main(argv=None)`, run as
`uv run python -m scripts.aux.<name>`. The collectors driving external calls
(collect_google_serp, collect_robots, collect_cited_urls,
collect_site_inventory, collect_directories) print a cost/volume estimate and
require `--confirm` before doing any network work.
"""
