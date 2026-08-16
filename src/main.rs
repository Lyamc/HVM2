#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

use clap::{Arg, ArgAction, ArgMatches, Command};
use ::hvm::{ast, cmp, hvm};
use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command as SysCommand;

#[cfg(feature = "c")]
extern "C" {
  fn hvm_c(book_buffer: *const u32);
  fn hvm_set_threads(n: u32);
}

#[cfg(feature = "cuda")]
extern "C" {
  fn hvm_cu(book_buffer: *const u32);
}

fn default_threads(_parallel: bool) -> u32 {
  hvm::env_threads_or(hvm::default_tpc(num_cpus::get_physical()))
}

fn threads_arg() -> Arg {
  Arg::new("threads")
    .long("threads")
    .short('t')
    .value_parser(clap::value_parser!(u32))
    .help("Worker threads. Default is min(8, 2^floor(log2(physical cores))); cap 16 via --threads / HVM_THREADS.")
}

fn cli_threads(sub: &ArgMatches, parallel: bool) -> u32 {
  sub.get_one::<u32>("threads").copied().unwrap_or_else(|| default_threads(parallel))
}

fn main() {
  let matches = Command::new("hvm")
    .about("HVM2: Higher-order Virtual Machine 2 (32-bit Version)")
    .version(env!("CARGO_PKG_VERSION"))
    .subcommand_required(true)
    .arg_required_else_help(true)
    .subcommand(
      Command::new("run")
        .about("Interprets a file (using Rust)")
        .arg(Arg::new("file").required(true))
        .arg(threads_arg()))
    .subcommand(
      Command::new("run-c")
        .about("Interprets a file (using C)")
        .arg(Arg::new("file").required(true))
        .arg(threads_arg())
        .arg(Arg::new("io")
          .long("io")
          .action(ArgAction::SetTrue)
          .help("Run with IO enabled"))
    )
    .subcommand(
      Command::new("run-cu")
        .about("Interprets a file (using CUDA)")
        .arg(Arg::new("file").required(true))
        .arg(Arg::new("io")
          .long("io")
          .action(ArgAction::SetTrue)
          .help("Run with IO enabled")))
    .subcommand(
      Command::new("run-wgpu")
        .alias("run-wg")
        .about("Interprets a file (using WebGPU / wgpu). Same .hvm books as `run`; no Bend changes.")
        .arg(Arg::new("file").required(true)))
    .subcommand(
      Command::new("gen-c")
        .about("Compiles a file with IO (to standalone C)")
        .arg(Arg::new("file").required(true))
        .arg(Arg::new("io")
          .long("io")
          .action(ArgAction::SetTrue)
          .help("Generate with IO enabled")))
    .subcommand(
      Command::new("gen-cu")
        .about("Compiles a file (to standalone CUDA)")
        .arg(Arg::new("file").required(true))
        .arg(Arg::new("io")
          .long("io")
          .action(ArgAction::SetTrue)
          .help("Generate with IO enabled")))
    .subcommand(
      Command::new("gen-rs")
        .about("Compiles a file to standalone Rust (compiled interact_call; rustc)")
        .arg(Arg::new("file").required(true)))
    .get_matches();

  match matches.subcommand() {
    Some(("run", sub_matches)) => {
      let file = sub_matches.get_one::<String>("file").expect("required");
      let threads = cli_threads(sub_matches, false);
      let code = fs::read_to_string(file).expect("Unable to read file");
      let book = ast::Book::parse(&code).unwrap_or_else(|er| panic!("{}",er)).build();
      run(&book, threads);
    }
    Some(("run-c", sub_matches)) => {
      let file = sub_matches.get_one::<String>("file").expect("required");
      let threads = cli_threads(sub_matches, true);
      let code = fs::read_to_string(file).expect("Unable to read file");
      let book = ast::Book::parse(&code).unwrap_or_else(|er| panic!("{}",er)).build();
      let mut data : Vec<u8> = Vec::new();
      book.to_buffer(&mut data);
      #[cfg(feature = "c")]
      unsafe {
        hvm_set_threads(threads);
        hvm_c(data.as_mut_ptr() as *mut u32);
      }
      #[cfg(not(feature = "c"))]
      println!("C runtime not available!\n");
    }
    Some(("run-cu", sub_matches)) => {
      let file = sub_matches.get_one::<String>("file").expect("required");
      let code = fs::read_to_string(file).expect("Unable to read file");
      let book = ast::Book::parse(&code).unwrap_or_else(|er| panic!("{}",er)).build();
      let mut data : Vec<u8> = Vec::new();
      book.to_buffer(&mut data);
      #[cfg(feature = "cuda")]
      unsafe {
        hvm_cu(data.as_mut_ptr() as *mut u32);
      }
      #[cfg(not(feature = "cuda"))]
      println!("CUDA runtime not available!\n If you've installed CUDA and nvcc after HVM, please reinstall HVM.");
    }
    Some(("run-wgpu", sub_matches)) => {
      let file = sub_matches.get_one::<String>("file").expect("required");
      let code = fs::read_to_string(file).expect("Unable to read file");
      let book = ast::Book::parse(&code).unwrap_or_else(|er| panic!("{}",er)).build();
      #[cfg(feature = "wgpu")]
      ::hvm::run_wgpu::run(&book);
      #[cfg(not(feature = "wgpu"))]
      println!("WebGPU runtime not available!\nRebuild with: cargo build --release --features wgpu");
    }
    Some(("gen-c", sub_matches)) => {
      // Reads book from file
      let file = sub_matches.get_one::<String>("file").expect("required");
      let code = fs::read_to_string(file).expect("Unable to read file");
      let book = ast::Book::parse(&code).unwrap_or_else(|er| panic!("{}",er)).build();

      // Gets optimal core count
      let cores = num_cpus::get_physical().max(1);
      let mut tpcl2 = (cores as f64).log2().floor() as u32;
      if tpcl2 > 4 { tpcl2 = 4; }

      // Generates the interpreted book
      let mut book_buf : Vec<u8> = Vec::new();
      book.to_buffer(&mut book_buf);
      let bookb = format!("{:?}", book_buf).replace("[","{").replace("]","}");
      let bookb = format!("static const u8 BOOK_BUF[] = {};", bookb);

      // Generates the C file
      let hvm_c = include_str!("hvm.c").replace("#include \"hvm_os.h\"", include_str!("hvm_os.h"));
      let hvm_c = format!("#define IO\n\n{hvm_c}");
      let hvm_c = hvm_c.replace("///COMPILED_INTERACT_CALL///", &cmp::compile_book(cmp::Target::C, &book));
      let hvm_c = hvm_c.replace("#define INTERPRETED", "#define COMPILED");
      let hvm_c = hvm_c.replace("//COMPILED_BOOK_BUF//", &bookb);
      let hvm_c = hvm_c.replace("#define WITHOUT_MAIN", "#define WITH_MAIN");
      let hvm_c = hvm_c.replace("#define TPC_L2 0", &format!("#define TPC_L2 {} // {} cores", tpcl2, cores));
      let hvm_c = format!("{hvm_c}\n\n{}", include_str!("run.c"));
      let hvm_c = hvm_c.replace(r#"#include "hvm.c""#, "");
      println!("{}", hvm_c);
    }
    Some(("gen-cu", sub_matches)) => {
      // Reads book from file
      let file = sub_matches.get_one::<String>("file").expect("required");
      let code = fs::read_to_string(file).expect("Unable to read file");
      let book = ast::Book::parse(&code).unwrap_or_else(|er| panic!("{}",er)).build();

      // Generates the interpreted book
      let mut book_buf : Vec<u8> = Vec::new();
      book.to_buffer(&mut book_buf);
      let bookb = format!("{:?}", book_buf).replace("[","{").replace("]","}");
      let bookb = format!("static const u8 BOOK_BUF[] = {};", bookb);

      //FIXME: currently, CUDA is faster on interpreted mode, so the compiler uses it.

      // Compile with compiled functions:
      //let hvm_c = include_str!("hvm.cu");
      //let hvm_c = hvm_c.replace("///COMPILED_INTERACT_CALL///", &cmp::compile_book(cmp::Target::CUDA, &book));
      //let hvm_c = hvm_c.replace("#define INTERPRETED", "#define COMPILED");
      
      // Generates the Cuda file
      let hvm_cu = include_str!("hvm.cu");
      let hvm_cu = format!("#define IO\n\n{hvm_cu}");
      let hvm_cu = hvm_cu.replace("//COMPILED_BOOK_BUF//", &bookb);
      let hvm_cu = hvm_cu.replace("#define WITHOUT_MAIN", "#define WITH_MAIN");
      let hvm_cu = format!("{hvm_cu}\n\n{}", include_str!("run.cu"));
      let hvm_cu = hvm_cu.replace("#include \"hvm_os.h\"", include_str!("hvm_os.h"));
      let hvm_cu = hvm_cu.replace(r#"#include "hvm.cu""#, "");
      println!("{}", hvm_cu);
    }
    Some(("gen-rs", sub_matches)) => {
      let file = sub_matches.get_one::<String>("file").expect("required");
      let code = fs::read_to_string(file).expect("Unable to read file");
      let book = ast::Book::parse(&code).unwrap_or_else(|er| panic!("{}",er)).build();
      print!("{}", ::hvm::gen::generate_rs(&book));
    }
    _ => unreachable!(),
  }
}

pub fn run(book: &hvm::Book, threads: u32) {
  // Initializes the global net
  let net = if threads > 1 {
    hvm::GNet::with_workers(1 << 29, 1 << 29, threads)
  } else {
    hvm::GNet::new(1 << 29, 1 << 29)
  };

  // Creates an initial redex that calls main
  let main_id = book.defs.iter().position(|def| def.name == "main").unwrap();
  let boot = hvm::Pair::new(hvm::Port::new(hvm::REF, main_id as u32), hvm::ROOT);
  net.vars_create(hvm::ROOT.get_val() as usize, hvm::NONE);

  // Starts the timer
  let start = std::time::Instant::now();

  // Evaluates
  if threads <= 1 {
    let mut tm = hvm::TMem::new(0, 1);
    tm.push_redex(&net, boot);
    tm.evaluator(&net, book);
  } else {
    hvm::TMem::evaluator_pool(&net, book, boot, threads);
  }
  
  // Stops the timer
  let duration = start.elapsed();

  //println!("{}", net.show());

  // Prints the result
  if let Some(tree) = ast::Net::readback(&net, book) {
    println!("Result: {}", tree.show());
  } else {
    println!("Readback failed. Printing GNet memdump...\n");
    println!("{}", net.show());
  }

  // Prints interactions and time
  let itrs = net.itrs.load(std::sync::atomic::Ordering::Relaxed);
  println!("- ITRS: {}", itrs);
  println!("- TIME: {:.2}s", duration.as_secs_f64());
  println!("- MIPS: {:.2}", itrs as f64 / duration.as_secs_f64() / 1_000_000.0);
}
