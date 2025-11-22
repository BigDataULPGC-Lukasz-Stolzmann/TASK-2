# Sparse Matrix-Vector Multiplication in Rust

Performance analysis of SpMV (Sparse Matrix-Vector Multiplication) using CSR format implemented in Rust.

## What is this?

This project implements and benchmarks sparse matrix-vector multiplication using the Compressed Sparse Row (CSR) format. It compares serial vs parallel performance using Rust's Rayon library.

## Quick Start

```bash
# Run the code
cd rust_spmv

# Matrix file mc2depi.mtx should be in the root directory

# Run benchmarks
cargo run --release
```

## Results

On Apple M3 Pro:

| Implementation | Time (ms) | GFLOP/s | Bandwidth (GB/s) | Speedup |
|---------------|-----------|---------|------------------|---------|
| Debug Serial | 72.3 | 0.058 | 0.494 | 1.0× |
| Debug Parallel | 11.6 | 0.364 | 3.092 | 6.3× |
| **Release Serial** | **1.4** | **2.989** | **25.4** | **51.5×** |
| **Release Parallel** | **0.75** | **5.630** | **47.9** | **97.0×** |

## Key Takeaways

- **Release mode is critical** - 50× performance difference from debug
- **Parallel speedup limited by memory bandwidth** - 1.9× in optimized code
- **Good memory bandwidth utilization** - 47.9 GB/s approaches hardware limits
- **Rust performs well** - competitive with C/C++ while staying memory-safe

## Project Structure

```
├── src/main.rs          # SpMV implementations and benchmarks
├── Cargo.toml           # Dependencies (sprs, rayon, rand)
├── paper/               # LaTeX paper and PDF
└── mc2depi.mtx          # Test matrix (525k×525k, 2.1M non-zeros)
```

## Dependencies

- `sprs` - Sparse matrix library for Rust
- `rayon` - Data parallelism library
- `rand` - Random number generation for test vectors

## Matrix Format

Uses Matrix Market format (.mtx files). The code automatically converts integer matrices to f64 for computation.

## How it works

1. **CSR Storage**: Stores only non-zero elements with column indices and row pointers
2. **Serial SpMV**: Simple row-wise computation `y[i] = sum(A[i,j] * x[j])`
3. **Parallel SpMV**: Uses Rayon to parallelize across matrix rows
4. **Benchmarking**: Measures time, calculates GFLOP/s and memory bandwidth

## Performance Notes

- Memory bandwidth bound operation (not compute bound)
- Cache-friendly row-wise access pattern
- Parallel efficiency limited by memory subsystem
- Compiler optimizations essential for good performance

## Using different matrices

Edit `src/main.rs` line 71 to change the matrix file:
```rust
let path = "your-matrix.mtx";
```

## Paper

Complete analysis available in `paper/sparse_matrix_spmv_analysis.pdf` with:
- Methodology details
- Performance analysis
- Comparison of debug vs release modes
- Discussion of results

