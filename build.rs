use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_nvcc() -> Option<PathBuf> {
  if Command::new("nvcc")
    .arg("--version")
    .stdout(std::process::Stdio::null())
    .stderr(std::process::Stdio::null())
    .status()
    .map(|s| s.success())
    .unwrap_or(false)
  {
    return Some(PathBuf::from("nvcc"));
  }
  let mut dirs = Vec::new();
  if let Ok(p) = env::var("CUDA_PATH").or_else(|_| env::var("CUDA_HOME")) {
    dirs.push(PathBuf::from(p));
  }
  dirs.push(PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"));
  dirs.push(PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"));
  dirs.push(PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"));
  dirs.push(PathBuf::from("/usr/local/cuda"));
  let exe = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
  for d in dirs {
    let cand = d.join("bin").join(exe);
    if cand.exists() {
      return Some(cand);
    }
  }
  None
}

fn main() {
  let logical = num_cpus::get().max(1);
  let physical = num_cpus::get_physical().max(1);
  // Cap compile-time TPC at 16 (2^4). 32 logical threads made a 4 GiB rbag
  // and oversubscribed steal on Windows. Override with HVM_TPC_L2.
  let mut tpcl2 = (physical as f64).log2().floor() as u32;
  if tpcl2 > 4 {
    tpcl2 = 4;
  }
  if let Ok(v) = env::var("HVM_TPC_L2") {
    if let Ok(n) = v.parse::<u32>() {
      tpcl2 = n.min(8);
    }
  }
  println!("cargo:rerun-if-env-changed=HVM_TPC_L2");
  println!("cargo:warning=C runtime TPC_L2={tpcl2} ({} workers, physical={physical}, logical={logical})", 1u32 << tpcl2);

  let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
  let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
  let target_family = env::var("CARGO_CFG_TARGET_FAMILY").unwrap_or_default();
  let is_windows = target_os == "windows";
  let is_windows_gnu = is_windows && (target_env == "gnu" || target_env == "gnullvm");

  println!("cargo:rerun-if-changed=src/run.c");
  println!("cargo:rerun-if-changed=src/hvm.c");
  println!("cargo:rerun-if-changed=src/hvm_os.h");
  println!("cargo:rerun-if-changed=src/run.cu");
  println!("cargo:rerun-if-changed=src/hvm.cu");
  println!("cargo:rerun-if-env-changed=HVM_SKIP_C");
  println!("cargo:rerun-if-env-changed=HVM_SKIP_CUDA");
  println!("cargo:rerun-if-env-changed=CUDA_PATH");
  println!("cargo:rerun-if-env-changed=CUDA_HOME");

  // Export dynamic symbols so IO plugins can resolve host functions.
  if target_family == "unix" {
    println!("cargo:rustc-link-arg=-rdynamic");
  } else if is_windows_gnu {
    println!("cargo:rustc-link-arg=-Wl,--export-all-symbols");
  }

  let skip_c = env::var_os("HVM_SKIP_C").is_some();
  if skip_c {
    println!("cargo:warning=HVM_SKIP_C set; the C runtime will not be compiled.");
  } else {
    let mut c = cc::Build::new();
    c.file("src/run.c")
      .opt_level(3)
      .warnings(false)
      .define("TPC_L2", &*tpcl2.to_string())
      .define("IO", None)
      .include("src");

    if c.get_compiler().is_like_msvc() {
      c.define("_CRT_SECURE_NO_WARNINGS", None);
      c.flag("/std:c11");
      c.flag("/experimental:c11atomics");
      c.flag("/Gy");
    } else {
      c.flag_if_supported("-std=c11");
      c.flag_if_supported("-ffunction-sections");
      c.flag_if_supported("-fdata-sections");
    }

    match c.try_compile("hvm-c") {
      Ok(_) => println!("cargo:rustc-cfg=feature=\"c\""),
      Err(e) => {
        println!("cargo:warning=WARNING: Failed to compile run.c: {e}");
        println!("cargo:warning=Ignoring run.c and proceeding. The C runtime will not be available.");
      }
    }
  }

  let skip_cuda = env::var_os("HVM_SKIP_CUDA").is_some();
  let nvcc = if skip_cuda { None } else { find_nvcc() };

  if let Some(nvcc) = nvcc {
    if let Some(bin) = Path::new(&nvcc).parent() {
      let old = env::var("PATH").unwrap_or_default();
      let sep = if is_windows { ";" } else { ":" };
      env::set_var("PATH", format!("{}{sep}{old}", bin.display()));
    }

    let cuda_root = nvcc
      .parent()
      .and_then(|p| p.parent())
      .map(|p| p.to_path_buf())
      .or_else(|| env::var("CUDA_PATH").or_else(|_| env::var("CUDA_HOME")).ok().map(PathBuf::from));

    if let Some(root) = &cuda_root {
      let lib = if is_windows {
        root.join("lib").join("x64")
      } else {
        root.join("lib64")
      };
      println!("cargo:rustc-link-search=native={}", lib.display());
    }

    let mut cu = cc::Build::new();
    cu.cuda(true)
      .file("src/run.cu")
      .define("IO", None)
      .include("src")
      .flag("-std=c++17")
      .flag("-diag-suppress=177")
      .flag("-diag-suppress=550")
      .flag("-diag-suppress=20039")
      // Native 30-series + PTX for later chips. 4090 (sm_89) can JIT the PTX.
      .flag("-gencode=arch=compute_86,code=sm_86")
      .flag("-gencode=arch=compute_86,code=compute_86")
      .flag("-allow-unsupported-compiler");

    match cu.try_compile("hvm-cu") {
      Ok(_) => {
        println!("cargo:rustc-cfg=feature=\"cuda\"");
        println!("cargo:warning=CUDA runtime compiled (nvcc={})", nvcc.display());
      }
      Err(e) => {
        println!("cargo:warning=WARNING: Failed to compile run.cu: {e}");
        println!("cargo:warning=The CUDA runtime will not be available.");
      }
    }
  } else if skip_cuda {
    println!("cargo:warning=HVM_SKIP_CUDA set; the CUDA runtime will not be compiled.");
  } else {
    println!("cargo:warning=WARNING: CUDA compiler not found. HVM will not be able to run on GPU.");
  }
}
