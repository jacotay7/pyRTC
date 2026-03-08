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
| PYWFS full loop | 58.1 kHz / 17.2 us | 4.5 kHz / 219.9 us | 26.6 kHz / 37.6 us | 4.6 kHz / 218.5 us | 270 Hz / 3703.4 us | 3.0 kHz / 335.2 us |
| SHWFS full loop | 78.2 kHz / 12.8 us | 5.1 kHz / 195.0 us | 26.6 kHz / 37.6 us | 5.1 kHz / 196.7 us | 268 Hz / 3730.7 us | 3.5 kHz / 289.7 us |
