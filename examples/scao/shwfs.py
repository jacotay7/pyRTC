"""Backward-compatible wrapper for the documented OOPAO soft-RTC example.

Historically this path existed as an example placeholder under ``examples/scao``.
The maintained script entry point now lives in ``run_soft_rtc.py``; this wrapper
keeps the older path usable for local workflows and notes.
"""

from run_soft_rtc import main


if __name__ == "__main__":
	raise SystemExit(main())
