import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from pyRTC import initExistingShm
from pyRTC.logging_utils import add_logging_cli_args, configure_logging_from_args


def rolling_average(data, window_size):
    n = len(data)
    if n < window_size:
        return []

    averages = []
    window_sum = sum(data[:window_size])
    averages.append(window_sum / window_size)

    for i in range(1, n - window_size + 1):
        window_sum += data[i + window_size - 1] - data[i - 1]
        averages.append(window_sum / window_size)

    return averages


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor a 1D shared memory signal in real-time.")
    parser.add_argument("shm", type=str, help="name of SHM to plot")
    parser.add_argument("--interval", type=float, default=0.1, help="seconds between updates")
    parser.add_argument("--window-size", type=int, default=10, help="rolling average window size")
    parser.add_argument("--max-size", type=int, default=1000, help="maximum samples to keep")
    add_logging_cli_args(parser)
    return parser


def main(argv=None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logger = configure_logging_from_args(args, app_name="pyrtc-shm-monitor", component_name=args.shm)

    shm_name = args.shm
    shm, _, _ = initExistingShm(shm_name)
    logger.info("Monitoring SHM %s interval=%s window_size=%s max_size=%s", shm_name, args.interval, args.window_size, args.max_size)

    update_interval = args.interval
    window_size = args.window_size
    max_size = args.max_size

    def compute_next_value():
        return np.max(shm.read_noblock())

    fig, ax = plt.subplots(figsize=(12, 5))
    (line,) = ax.plot([], [], lw=2)

    past_values = [compute_next_value()] * window_size

    ax.set_xlim(0, max_size)
    ax.set_ylabel(shm_name, size=16)
    ax.set_xlabel("Time [arb]", size=16)
    ax.grid()

    plt.ion()
    plt.show()

    while True:
        if len(past_values) >= max_size:
            past_values[:-1] = past_values[1:]
            past_values[-1] = compute_next_value()
        else:
            past_values.append(compute_next_value())

        ydata = rolling_average(past_values, window_size)
        xdata = list(range(len(ydata)))
        line.set_data(xdata, ydata)
        ax.set_ylim(np.percentile(ydata, 5), np.percentile(ydata, 95))
        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(update_interval)


if __name__ == "__main__":
    raise SystemExit(main())
