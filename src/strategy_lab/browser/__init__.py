"""The research browser's one page.

A *view*, not a record. `strategy-lab serve` hosts the frozen per-run
``plot.html`` that a backtest wrote and will re-render byte-identically; this
page recomputes from stored candles on every request and stores nothing. Two
commands, and they must not become one.
"""
