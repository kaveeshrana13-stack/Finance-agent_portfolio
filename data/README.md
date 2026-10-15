# data/

`nse_equity_list.csv` is NSE's official list of listed equities
(<https://archives.nseindia.com/content/equities/EQUITY_L.csv>), downloaded 2026-10-15.

It is the source of truth for symbol resolution (`tools/resolver.py`). Symbols change when
companies rename, merge or demerge, so refresh it periodically:

```python
from tools.resolver import refresh_nse_list
print(refresh_nse_list())
```

If NSE blocks the download, the existing local copy keeps working.
