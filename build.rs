fn main() {
  let cores = num_cpus::get();
  let tpcl2 = (cores as f64).log2().floor() as u32;

  println!("cargo:rerun-if-changed=src/run.c");
  println!("cargo:rerun-if-changed=src/hvm.c");
  println!("cargo:rerun-if-changed=src/hvm_os.h");
  println!("cargo:rerun-if-changed=src/run.cu");
  println!("cargo:rerun-if-changed=src/hvm.cu");

  // GNU ld only. MSVC / lld-link reject -rdynamic.
  if cfg!(unix) {
    println!("cargo:rustc-link-arg=-rdynamic");
  }

  let mut c = cc::Build::new();
  c.file("src/run.c")
    .opt_level(3)
    .warnings(false)
    .define("TPC_L2", &*tpcl2.to_string())
    .define("IO", None)
    .include("src");

  if cfg!(windows) {
    c.define("_CRT_SECURE_NO_WARNINGS", None);
    if c.get_compiler().is_like_msvc() {
      // C11 _Atomic in hvm.c: VS 2022 needs both flags.
      c.flag("/std:c11");
      c.flag("/experimental:c11atomics");
    } else {
      c.flag_if_supported("-std=c11");
    }
  }

  match c.try_compile("hvm-c") {
    Ok(_) => println!("cargo:rustc-cfg=feature=\"c\""),
    Err(e) => {
      println!("cargo:warning=WARNING: Failed to compile run.c: {e}");
      println!("cargo:warning=Ignoring run.c and proceeding. The C runtime will not be available.");
    }
  }

  if std::process::Command::new("nvcc")
    .arg("--version")
    .stdout(std::process::Stdio::null())
    .stderr(std::process::Stdio::null())
    .status()
    .is_ok()
  {
    let cuda_lib = if let Ok(cuda_path) = std::env::var("CUDA_PATH")
      .or_else(|_| std::env::var("CUDA_HOME"))
    {
      if cfg!(windows) {
        format!("{}/lib/x64", cuda_path)
      } else {
        format!("{}/lib64", cuda_path)
      }
    } else if cfg!(windows) {
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
  } else {
    println!("cargo:warning=WARNING: CUDA compiler not found. HVM will not be able to run on GPU.");
  }
}
