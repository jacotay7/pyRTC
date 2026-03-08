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

### Core Compute Benchmarks

Values are reported as `p99 throughput / p99 latency`.

| Kernel | 10x10 CPU | 10x10 GPU | 20x20 CPU | 20x20 GPU | 60x60 CPU | 60x60 GPU |
| --- | --- | --- | --- | --- | --- | --- |
| WFS downsample | 1587.2 kHz / 0.6 us | - | 861.8 kHz / 1.2 us | - | 159.7 kHz / 6.3 us | - |
| WFS rotate | 628.9 kHz / 1.6 us | - | 289.8 kHz / 3.5 us | - | 38.8 kHz / 25.8 us | - |
| WFC modal->zonal | 1063.5 kHz / 0.9 us | - | 186.9 kHz / 5.4 us | - | 1.4 kHz / 729.2 us | - |
| Loop leaky integrator | 763.1 kHz / 1.3 us | 28.4 kHz / 35.3 us | 82.4 kHz / 12.1 us | 29.6 kHz / 33.8 us | 543 Hz / 1841.9 us | 11.6 kHz / 86.3 us |
| PYWFS slopes | 546.4 kHz / 1.8 us | 8.7 kHz / 114.6 us | 222.7 kHz / 4.5 us | 8.6 kHz / 116.0 us | 30.6 kHz / 32.7 us | 7.7 kHz / 129.2 us |
| SHWFS slopes | 625.0 kHz / 1.6 us | - | 432.9 kHz / 2.3 us | - | 64.5 kHz / 15.5 us | - |
