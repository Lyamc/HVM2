use std::env;

fn main() {
  let cores = num_cpus::get();
  let tpcl2 = (cores as f64).log2().floor() as u32;

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
  // GNU ld: -rdynamic. MinGW: --export-all-symbols. MSVC exports via .def / dllexport.
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
      // VS 2022: C11 _Atomic lives behind both flags.
      c.define("_CRT_SECURE_NO_WARNINGS", None);
      c.flag("/std:c11");
      c.flag("/experimental:c11atomics");
      c.flag("/Gy"); // function-level linking, pairs with /OPT:REF
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
  let nvcc_ok = !skip_cuda
    && std::process::Command::new("nvcc")
      .arg("--version")
      .stdout(std::process::Stdio::null())
      .stderr(std::process::Stdio::null())
      .status()
      .map(|s| s.success())
      .unwrap_or(false);

  if nvcc_ok {
    let cuda_lib = if let Ok(cuda_path) = env::var("CUDA_PATH").or_else(|_| env::var("CUDA_HOME")) {
      if is_windows {
        format!("{}/lib/x64", cuda_path)
      } else {
        format!("{}/lib64", cuda_path)
      }
    } else if is_windows {
      "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/lib/x64".to_string()
    } else {
      "/usr/local/cuda/lib64".to_string()
    };
    println!("cargo:rustc-link-search=native={cuda_lib}");

    cc::Build::new()
      .cuda(true)
      .file("src/run.cu")
      .define("IO", None)
      .flag("-diag-suppress=177")
      .flag("-diag-suppress=550")
      .flag("-diag-suppress=20039")
      .compile("hvm-cu");

    println!("cargo:rustc-cfg=feature=\"cuda\"");
  } else if skip_cuda {
    println!("cargo:warning=HVM_SKIP_CUDA set; the CUDA runtime will not be compiled.");
  } else {
    println!("cargo:warning=WARNING: CUDA compiler not found. HVM will not be able to run on GPU.");
  }
}
