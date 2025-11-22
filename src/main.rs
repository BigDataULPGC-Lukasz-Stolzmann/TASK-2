use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use sprs::io::read_matrix_market;
use sprs::CsMat;
use sprs::TriMat;
use std::time::Instant;

fn spmv_serial(A: &CsMat<f64>, x: &[f64]) -> Vec<f64> {
    let nrows = A.rows();
    let mut y = vec![0.0f64; nrows];

    for i in 0..nrows {
        let mut sum = 0.0;
        if let Some(row) = A.outer_view(i) {
            for (col, val) in row.iter() {
                sum += val * x[col];
            }
        }
        y[i] = sum;
    }
    y
}

fn spmv_parallel(A: &CsMat<f64>, x: &[f64]) -> Vec<f64> {
    let nrows = A.rows();

    (0..nrows)
        .into_par_iter()
        .map(|i| {
            let mut sum = 0.0;
            if let Some(row) = A.outer_view(i) {
                for (col, val) in row.iter() {
                    sum += val * x[col];
                }
            }
            sum
        })
        .collect()
}

fn benchmark<F>(name: &str, A: &CsMat<f64>, x: &[f64], f: F)
where
    F: Fn(&CsMat<f64>, &[f64]) -> Vec<f64>,
{
    println!("\n[INFO] Benchmark: {name}");

    let _ = f(A, x);

    let start = Instant::now();
    let _y = f(A, x);
    let dt = start.elapsed().as_secs_f64();

    let nnz = A.nnz() as f64;

    let gflops = (2.0 * nnz) / (dt * 1.0e9);

    let n = A.rows() as f64;
    let bytes_values = 8.0 * nnz;
    let bytes_indices = 4.0 * nnz;
    let bytes_indptr = 4.0 * (A.indptr().len() as f64);
    let bytes_x = 8.0 * n;
    let bytes_y = 8.0 * n;

    let total_bytes = bytes_values + bytes_indices + bytes_indptr + bytes_x + bytes_y;
    let bandwith_gb_s = total_bytes / (dt * 1.0e9);

    println!("{name}: time = {dt:.6} s, perf = {gflops:.3} GFLOP/s, BW = {bandwith_gb_s:.3} GB/s");
}

fn main() {
    let path = "./mc2depi.mtx";

    println!("[INFO] Loading matrix from {path} ...");
    let tri_mat: TriMat<i32> = read_matrix_market(path).expect("Failed to load Matrix Market file");
    let A = tri_mat.to_csr().map(|&x| x as f64);
    println!(
        "[INFO] Matrix: {} x {} (nnz = {})",
        A.rows(),
        A.cols(),
        A.nnz()
    );

    let mut rng = StdRng::seed_from_u64(42);
    let x: Vec<f64> = (0..A.cols()).map(|_| rng.r#gen::<f64>()).collect();

    benchmark("Rust CSR serial", &A, &x, spmv_serial);
    benchmark("Rust CSR parallel (Rayon)", &A, &x, spmv_parallel);
}
