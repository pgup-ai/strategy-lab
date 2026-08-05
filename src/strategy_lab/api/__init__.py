"""The read-only research browser's server side.

Nothing here writes to ``signals``, to ``reports/`` or to ``market_candles``
except through the fetch path ``server.py`` already owns.
"""
