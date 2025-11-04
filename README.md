# Sobel Filter Implementation - CSC 746 Project 5

## Author
Binrong Zhu

## Build Instructions

### Environment Setup (on Perlmutter)
```bash
module load PrgEnv-nvidia
export CC=cc
export CXX=CC
```

### Compilation
```bash
mkdir build
cd build
cmake ../ -Wno-dev
make
```

## Run Instructions

### CPU Version
```bash
cd build
export OMP_NUM_THREADS=16  # Set thread count
./sobel_cpu
```

### CUDA Version
```bash
cd build
./sobel_gpu <num_blocks> <threads_per_block>

# Examples:
./sobel_gpu 1024 128    # Best configuration
./sobel_gpu 256 512
```

### OpenMP Offload Version
```bash
cd build
./sobel_cpu_omp_offload
```

## File Structure
```
sobel-harness-instructional/
├── sobel_cpu.cpp                # CPU implementation
├── sobel_gpu.cu                 # CUDA implementation  
├── sobel_cpu_omp_offload.cpp    # OpenMP offload implementation
├── CMakeLists.txt               # Build configuration
├── README.md                    # This file
├── generate_plots.py            # Script to generate performance plots
├── show_sobel.py                # Script to visualize Sobel output
├── data/                        # Input data directory
│   └── zebra-gray-int8-4x       # Input image (7112×5146 pixels)
├── scripts/                     # Additional scripts
│   └── imshow.py                # Image visualization script
├── build/                       # Build directory (created by cmake)
│   ├── sobel_cpu                # CPU executable
│   ├── sobel_gpu                # CUDA executable
│   └── sobel_cpu_omp_offload    # OpenMP offload executable
└── sobel_results/               # Results directory
    ├── processed-raw-int8-4x-cpu.dat
    ├── processed-raw-int8-4x-gpu.dat
    └── processed-raw-int8-4x-cpu-omp-offload.dat
```

## Input/Output

**Input:**
- File: `data/zebra-gray-int8-4x`
- Size: 7112 × 5146 pixels
- Format: 8-bit grayscale raw data

**Output:**
- CPU output: `data/processed-raw-int8-4x-cpu.dat`
- CUDA output: `data/processed-raw-int8-4x-gpu.dat`
- OpenMP offload output: `data/processed-raw-int8-4x-cpu-omp-offload.dat`

## Visualizing Results
```bash
module load python
python scripts/imshow.py data/zebra-gray-int8-4x 7112 5146
python scripts/imshow.py data/processed-raw-int8-4x-cpu.dat 7112 5146
```

## Performance Data Files

Performance test results are saved in:
- `all_cuda_results.txt` - Runtime for all 42 CUDA configurations
- `complete_ncu_data.csv` - NCU profiling data (occupancy, bandwidth)
- `figure*.png` - Performance visualization plots

## Notes

**Best CUDA Configuration:** 1024 blocks × 128 threads
- Runtime: ~0.008 seconds
- Achieves 2.6× speedup over 16-thread CPU

**Hardware:** NERSC Perlmutter with NVIDIA A100 GPU