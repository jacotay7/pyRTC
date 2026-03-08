### Benchmark Host

| Component | Value |
| --- | --- |
| CPU | AMD Ryzen 9 9950X3D 16-Core Processor |
| CPU Threads | 32 |
| GPU | NVIDIA GeForce RTX 5090 |
| GPU Memory | 32607 MiB |
| NVIDIA Driver | 580.126.09 |
| Python | 3.12.0 |
| Torch | 2.10.0+cu128 |
| CUDA | 12.8 |

### Synthetic AO Loop Benchmarks

Values are reported as `p99 throughput / p99 latency`.

| Loop | 10x10 CPU | 10x10 GPU | 20x20 CPU | 20x20 GPU | 60x60 CPU | 60x60 GPU |
| --- | --- | --- | --- | --- | --- | --- |
| PYWFS full loop | 82.8 kHz / 12.1 us | 4.3 kHz / 230.3 us | 24.5 kHz / 40.7 us | 4.3 kHz / 232.8 us | 263 Hz / 3806.8 us | 2.9 kHz / 339.8 us |
| SHWFS full loop | 68.7 kHz / 14.6 us | 4.5 kHz / 222.2 us | 24.8 kHz / 40.2 us | 4.7 kHz / 213.4 us | 262 Hz / 3820.0 us | 3.3 kHz / 301.6 us |
