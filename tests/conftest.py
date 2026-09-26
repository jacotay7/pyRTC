import os

# Tests must never open GUI windows: headless CI runners (notably Windows with
# Tk) abort when a figure is shown. Set before anything imports pyplot.
# Workaround for library code calling plt.show(): see jacotay7/pyRTC#34.
os.environ.setdefault("MPLBACKEND", "Agg")

from testsupport import unique_name as _unique_name
from testsupport import unlink_private_streams as _unlink_private_streams


unique_name = _unique_name
unlink_private_streams = _unlink_private_streams
