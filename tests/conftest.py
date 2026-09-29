import os

# Tests must never open GUI windows, and some library code still calls
# plt.show() (jacotay7/pyRTC#34). Set before anything imports pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")

from testsupport import unique_name as _unique_name
from testsupport import pyrtc_logs_propagate as _pyrtc_logs_propagate
from testsupport import unlink_private_streams as _unlink_private_streams


unique_name = _unique_name
unlink_private_streams = _unlink_private_streams
pyrtc_logs_propagate = _pyrtc_logs_propagate
