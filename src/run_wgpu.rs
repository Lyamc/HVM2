//! WebGPU interpreter for existing HVM books.
//!
//! Same `ast::Book` / `.hvm` encoding as `hvm run`. Bend programs are not
//! rewritten: `ROOT` (`0xFFFFFFF8`) is remapped to `vars[0]` only inside this
//! backend so a 2^16 heap can host it. Readback uses `ast::Tree::readback`.

use crate::ast;
use crate::hvm;
use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use wgpu::util::DeviceExt;

const WORKER_RECYCLE: i32 = 10;
const SUPERVISOR_EPOCH: i32 = 11;
const WORKERS_PER_EPOCH: u32 = 3;

const SHADER: &str = include_str!("shaders/hvm.wgsl");

const NONE: u32 = 0xFFFFFFFF;
const ROOT: u32 = 0xFFFFFFF8;

const M_ITRS: usize = 0;
const M_OOM: usize = 1;
const M_ERR: usize = 2;
const M_NLEN: usize = 3;
const M_VLEN: usize = 4;
const M_MAX: usize = 5;
const M_NTHREADS: usize = 6;
const M_RSPAN: usize = 7;
const M_RLEN: usize = 8;
const M_NROVER: usize = 9;
const M_VROVER: usize = 10;
const M_OFLOW: usize = 11;
const M_OFLOW_BASE: usize = 12;
const M_OFLOW_CAP: usize = 13;
const META_WORDS: usize = 16;

const W_NPUT: usize = 0;
const W_VPUT: usize = 1;
const W_NWRAP: usize = 2;
const W_VWRAP: usize = 3;
const W_RLEN: usize = 4;
const W_RPUT: usize = 6;
const W_STRIDE: usize = 8;

const MAX_DEF_SLOTS: usize = 1024;
const DEFAULT_HEAP: u32 = 1 << 23; // 8M nodes (~64 MiB pairs); smaller working set than 32M
const WG: u32 = 64;
const DEFAULT_NTHREADS: u32 = 8; // sieve keeps a few live redexes; HVM_WGPU_THREADS=1 drops node atomics
const DEFAULT_RSPAN: u32 = 1024;
const DEFAULT_OVERFLOW: u32 = 1 << 22;
const MAX_TURNS: u32 = 1 << 20;
const STEPS_PER_DISPATCH: u32 = 4096; // 32768 TDRs on a fragmented resume heap
/// This adapter/driver dies after ~256 compute dispatches on one device.
const KERNELS_PER_SUBMIT: u32 = 1;
const RECYCLE_AFTER_KERNELS: u32 = 40;
const CHECK_EVERY: u32 = 16;
const MAX_WORKER_RETRIES: u32 = 3;

pub fn run(book: &hvm::Book) {
  if env::var("HVM_WGPU_WORKER").ok().as_deref() == Some("1") {
    match run_inner(book) {
      Ok(true) => std::process::exit(0),
      Ok(false) => std::process::exit(WORKER_RECYCLE),
      Err(e) => {
        eprintln!("run-wgpu: {e}");
        std::process::exit(1);
      }
    }
  }
  if env::var("HVM_WGPU_SUPERVISOR").ok().as_deref() == Some("1") {
    match supervise() {
      Ok(()) => std::process::exit(0),
      Err(e) => {
        eprintln!("run-wgpu: {e}");
        std::process::exit(1);
      }
    }
  }
  match supervise_outer() {
    Ok(()) => {}
    Err(e) => {
      eprintln!("run-wgpu: {e}");
      std::process::exit(1);
    }
  }
}

fn supervise_outer() -> Result<(), String> {
  let exe = env::current_exe().map_err(|e| e.to_string())?;
  let snap = env::var("HVM_WGPU_SNAP")
    .map(PathBuf::from)
    .unwrap_or_else(|_| env::temp_dir().join(format!("hvm-wgpu-{}", std::process::id())));
  std::fs::create_dir_all(&snap).map_err(|e| e.to_string())?;
  let args: Vec<String> = env::args().skip(1).collect();
  let debug = env::var("HVM_WGPU_DEBUG").ok().as_deref() == Some("1");
  let recycle_ms = env::var("HVM_WGPU_RECYCLE_MS")
    .ok()
    .and_then(|s| s.parse().ok())
    .unwrap_or(2_000u64);
  let mut epochs = 0u32;
  let mut lost = 0u32;
  loop {
    let code = spawn_supervisor_epoch(&exe, &args, &snap, debug)?;
    match Some(code) {
      Some(0) => {
        let _ = std::fs::remove_dir_all(&snap);
        return Ok(());
      }
      Some(SUPERVISOR_EPOCH) => {
        epochs += 1;
        lost = 0;
        if debug {
          eprintln!("wgpu outer: epoch #{epochs}, rest {recycle_ms}ms");
        }
        std::thread::sleep(std::time::Duration::from_millis(recycle_ms));
      }
      Some(101) | Some(1) if snap.join("node.bin").exists() => {
        lost += 1;
        if lost > 3 {
          return Err(format!(
            "device-lost {lost} times in a row at snap {}; giving up",
            snap.display()
          ));
        }
        let wait = 30_000u64.saturating_mul(lost as u64);
        if debug {
          eprintln!("wgpu outer: device-lost #{lost}, rest {wait}ms");
        }
        std::thread::sleep(std::time::Duration::from_millis(wait));
      }
      other => {
        let _ = std::fs::remove_dir_all(&snap);
        return Err(format!("supervisor exited {other:?}"));
      }
    }
  }
}

fn supervise() -> Result<(), String> {
  let exe = env::current_exe().map_err(|e| e.to_string())?;
  let snap = env::var("HVM_WGPU_SNAP")
    .map(PathBuf::from)
    .unwrap_or_else(|_| env::temp_dir().join(format!("hvm-wgpu-{}", std::process::id())));
  std::fs::create_dir_all(&snap).map_err(|e| e.to_string())?;
  let args: Vec<String> = env::args().skip(1).collect();
  let debug = env::var("HVM_WGPU_DEBUG").ok().as_deref() == Some("1");
  let mut rounds = 0u32;
  let mut retries = 0u32;
  let backend = env::var("HVM_WGPU_BACKEND").unwrap_or_else(|_| "vulkan".into());
  let recycle_ms = env::var("HVM_WGPU_RECYCLE_MS")
    .ok()
    .and_then(|s| s.parse().ok())
    .unwrap_or(8_000u64);
  let lost_ms = env::var("HVM_WGPU_COOLDOWN_MS")
    .ok()
    .and_then(|s| s.parse().ok())
    .unwrap_or(8_000u64);
  loop {
    let mut cmd = Command::new(&exe);
    apply_gpu_isolation(&mut cmd);
    cmd.args(&args)
      .env("HVM_WGPU_WORKER", "1")
      .env("HVM_WGPU_SNAP", &snap)
      .env("HVM_WGPU_BACKEND", &backend)
      .env_remove("HVM_WGPU_SUPERVISOR")
      .stdin(Stdio::inherit())
      .stdout(Stdio::inherit())
      .stderr(Stdio::inherit());
    let status = cmd.status().map_err(|e| format!("spawn worker: {e}"))?;
    match status.code() {
      Some(0) => return Ok(()),
      Some(WORKER_RECYCLE) => {
        rounds += 1;
        retries = 0;
        if rounds >= WORKERS_PER_EPOCH {
          if debug {
            eprintln!("wgpu supervisor: epoch done after {rounds} workers");
          }
          std::process::exit(SUPERVISOR_EPOCH);
        }
        if debug {
          eprintln!("wgpu supervisor: worker recycle #{rounds}, rest {recycle_ms}ms");
        }
        std::thread::sleep(std::time::Duration::from_millis(recycle_ms));
      }
      Some(101) | None => {
        // Device-lost: end this process tree so the next epoch can start
        // outside it. Retrying here always dies at the same ITRS.
        if snap.join("node.bin").exists() {
          if debug {
            eprintln!("wgpu supervisor: device-lost with snap, exit 101");
          }
          // Must be 101, not SUPERVISOR_EPOCH (11): the outer loop treats 11
          // as a healthy recycle and only waits 2s, which storms a lost GPU.
          std::process::exit(101);
        }
        retries += 1;
        if retries > MAX_WORKER_RETRIES {
          let _ = std::fs::remove_dir_all(&snap);
          return Err(format!(
            "worker exited {:?} after {retries} retr{}.",
            status.code(),
            if retries == 1 { "y" } else { "ies" }
          ));
        }
        // DX12 on this NVIDIA adapter has no SHADER_INT64_ATOMIC_ALL_OPS.
        // Stay on Vulkan and wait out WDDM/TDR instead.
        let wait = lost_ms.saturating_mul(retries as u64).min(300_000);
        if debug {
          eprintln!(
            "wgpu supervisor: device-lost (exit {:?}), retry {retries}/{MAX_WORKER_RETRIES} after {wait}ms (backend={backend})",
            status.code()
          );
        }
        maybe_gpu_reset(debug);
        std::thread::sleep(std::time::Duration::from_millis(wait));
      }
      Some(1) if snap.join("node.bin").exists() => {
        retries += 1;
        if retries > MAX_WORKER_RETRIES {
          let _ = std::fs::remove_dir_all(&snap);
          return Err(format!("worker exited Some(1) after {retries} retries"));
        }
        if debug {
          eprintln!("wgpu supervisor: worker error 1 with snap, retry {retries} after {lost_ms}ms");
        }
        std::thread::sleep(std::time::Duration::from_millis(lost_ms));
      }
      other => {
        let _ = std::fs::remove_dir_all(&snap);
        return Err(format!("worker exited {other:?}"));
      }
    }
  }
}

#[cfg(windows)]
fn spawn_epoch_loop_wmi(
  exe: &Path,
  args: &[String],
  snap: &Path,
  debug: bool,
  recycle_ms: u64,
) -> Result<i32, String> {
  let loop_cmd = snap.join("loop_epochs.cmd");
  let done_txt = snap.join("done.txt");
  let log_txt = snap.join("epoch.log");
  let wait_ps1 = snap.join("wait_loop.ps1");
  let _ = std::fs::remove_file(&done_txt);
  let rest_s = (recycle_ms / 1000).max(1);
  let mut bat = String::new();
  bat.push_str("@echo off\r\nsetlocal\r\n");
  bat.push_str("set HVM_WGPU_SUPERVISOR=1\r\nset HVM_WGPU_WORKER=\r\n");
  bat.push_str(&format!("set \"HVM_WGPU_SNAP={}\"\r\n", snap.display()));
  if debug {
    bat.push_str("set HVM_WGPU_DEBUG=1\r\n");
  }
  bat.push_str(":loop\r\n");
  bat.push_str(&format!("\"{}\"", exe.display()));
  for a in args {
    bat.push_str(&format!(" \"{a}\""));
  }
  bat.push_str(&format!(" >>\"{}\" 2>&1\r\nset EC=%ERRORLEVEL%\r\n", log_txt.display()));
  bat.push_str("if %EC%==0 (\r\n>\"%HVM_WGPU_SNAP%\\done.txt\" echo 0\r\nexit /b 0\r\n)\r\n");
  bat.push_str(&format!(
    "if %EC%=={SUPERVISOR_EPOCH} (\r\nping -n {ping_n} 127.0.0.1 >nul\r\ngoto loop\r\n)\r\n",
    ping_n = rest_s + 1,
  ));
  bat.push_str(">\"%HVM_WGPU_SNAP%\\done.txt\" echo %EC%\r\nexit /b %EC%\r\n");
  std::fs::write(&loop_cmd, &bat).map_err(|e| e.to_string())?;
  let ps = format!(
    "$done = '{}'\r\n$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{ CommandLine = 'cmd.exe /d /c \"{}\"' }}\r\nif ($null -eq $r -or $r.ReturnValue -ne 0) {{ Set-Content -Path $done -Value 90; exit 90 }}\r\n$id = $r.ProcessId\r\n$deadline = (Get-Date).AddHours(6)\r\nwhile (-not (Test-Path -LiteralPath $done)) {{\r\n  if ((Get-Date) -gt $deadline) {{ Set-Content -Path $done -Value 92; break }}\r\n  if ($id -and -not (Get-Process -Id $id -ErrorAction SilentlyContinue)) {{\r\n    Start-Sleep -Seconds 2\r\n    if (-not (Test-Path -LiteralPath $done)) {{ Set-Content -Path $done -Value 91; break }}\r\n  }}\r\n  Start-Sleep -Milliseconds 800\r\n}}\r\n",
    done_txt.display().to_string().replace('\'', "''"),
    loop_cmd.display().to_string().replace('\'', "''"),
  );
  std::fs::write(&wait_ps1, ps).map_err(|e| e.to_string())?;
  if debug {
    eprintln!("wgpu outer: WMI epoch-loop {}", loop_cmd.display());
  }
  let _ = Command::new("powershell")
    .args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-File"])
    .arg(&wait_ps1)
    .status()
    .map_err(|e| format!("powershell: {e}"))?;
  if let Ok(log) = std::fs::read_to_string(&log_txt) {
    if !log.is_empty() {
      eprint!("{log}");
    }
  }
  if let Ok(text) = std::fs::read_to_string(&done_txt) {
    if let Ok(code) = text.trim().parse::<i32>() {
      return Ok(code);
    }
  }
  Err("epoch loop produced no done.txt".into())
}

fn spawn_supervisor_epoch(
  exe: &Path,
  args: &[String],
  snap: &Path,
  debug: bool,
) -> Result<i32, String> {
  #[cfg(windows)]
  if env::var("HVM_WGPU_WMI").ok().as_deref() == Some("1") {
    match spawn_supervisor_wmi(exe, args, snap, debug) {
      Ok(code) => return Ok(code),
      Err(e) => {
        if debug {
          eprintln!("wgpu outer: WMI spawn failed ({e}), falling back to child process");
        }
      }
    }
  }
  let mut cmd = Command::new(exe);
  apply_gpu_isolation(&mut cmd);
  cmd.args(args)
    .env("HVM_WGPU_SUPERVISOR", "1")
    .env("HVM_WGPU_SNAP", snap)
    .env_remove("HVM_WGPU_WORKER")
    .stdin(Stdio::inherit())
    .stdout(Stdio::inherit())
    .stderr(Stdio::inherit());
  Ok(cmd.status().map_err(|e| format!("spawn supervisor: {e}"))?.code().unwrap_or(1))
}

#[cfg(windows)]
fn spawn_supervisor_wmi(
  exe: &Path,
  args: &[String],
  snap: &Path,
  debug: bool,
) -> Result<i32, String> {
  let run_cmd = snap.join("run_epoch.cmd");
  let wait_ps1 = snap.join("wait_epoch.ps1");
  let exit_txt = snap.join("exit.txt");
  let log_txt = snap.join("epoch.log");
  let _ = std::fs::remove_file(&exit_txt);
  let _ = std::fs::remove_file(&log_txt);
  let mut bat = String::new();
  bat.push_str("@echo off\r\n");
  bat.push_str("set HVM_WGPU_SUPERVISOR=1\r\n");
  bat.push_str("set HVM_WGPU_WORKER=\r\n");
  bat.push_str(&format!("set \"HVM_WGPU_SNAP={}\"\r\n", snap.display()));
  if debug {
    bat.push_str("set HVM_WGPU_DEBUG=1\r\n");
  }
  if let Ok(v) = env::var("HVM_WGPU_BACKEND") {
    bat.push_str(&format!("set \"HVM_WGPU_BACKEND={v}\"\r\n"));
  }
  if let Ok(v) = env::var("HVM_WGPU_HEAP") {
    bat.push_str(&format!("set \"HVM_WGPU_HEAP={v}\"\r\n"));
  }
  if let Ok(v) = env::var("HVM_WGPU_THREADS") {
    bat.push_str(&format!("set \"HVM_WGPU_THREADS={v}\"\r\n"));
  }
  if let Ok(v) = env::var("HVM_WGPU_STEPS") {
    bat.push_str(&format!("set \"HVM_WGPU_STEPS={v}\"\r\n"));
  }
  bat.push_str(&format!("\"{}\"", exe.display()));
  for a in args {
    bat.push_str(&format!(" \"{a}\""));
  }
  bat.push_str(&format!(
    " >>\"{}\" 2>&1\r\n>\"{}\" echo %ERRORLEVEL%\r\n",
    log_txt.display(),
    exit_txt.display()
  ));
  std::fs::write(&run_cmd, bat).map_err(|e| e.to_string())?;
  let ps = format!(
    "$exit = '{}'\r\n$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{ CommandLine = 'cmd.exe /d /c \"{}\"' }}\r\nif ($null -eq $r -or $r.ReturnValue -ne 0) {{ Set-Content -Path $exit -Value 90; exit 90 }}\r\n$id = $r.ProcessId\r\n$deadline = (Get-Date).AddHours(2)\r\nwhile (-not (Test-Path -LiteralPath $exit)) {{\r\n  if ((Get-Date) -gt $deadline) {{ Set-Content -Path $exit -Value 92; break }}\r\n  if ($id -and -not (Get-Process -Id $id -ErrorAction SilentlyContinue)) {{\r\n    Start-Sleep -Seconds 1\r\n    if (-not (Test-Path -LiteralPath $exit)) {{ Set-Content -Path $exit -Value 91; break }}\r\n  }}\r\n  Start-Sleep -Milliseconds 400\r\n}}\r\n",
    exit_txt.display().to_string().replace('\'', "''"),
    run_cmd.display().to_string().replace('\'', "''"),
  );
  std::fs::write(&wait_ps1, ps).map_err(|e| e.to_string())?;
  if debug {
    eprintln!("wgpu outer: WMI-spawn {}", run_cmd.display());
  }
  let st = Command::new("powershell")
    .args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-File"])
    .arg(&wait_ps1)
    .status()
    .map_err(|e| format!("powershell: {e}"))?;
  if let Ok(log) = std::fs::read_to_string(&log_txt) {
    if !log.is_empty() {
      eprint!("{log}");
    }
  }
  if let Ok(text) = std::fs::read_to_string(&exit_txt) {
    if let Ok(code) = text.trim().parse::<i32>() {
      return Ok(code);
    }
  }
  Err(format!(
    "WMI supervisor produced no exit code (powershell {})",
    st.code().unwrap_or(-1)
  ))
}

fn apply_gpu_isolation(cmd: &mut Command) {
  #[cfg(windows)]
  {
    use std::os::windows::process::CommandExt;
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x00000200;
    cmd.creation_flags(CREATE_NEW_PROCESS_GROUP);
  }
}

fn maybe_gpu_reset(debug: bool) {
  if env::var("HVM_WGPU_GPU_RESET").ok().as_deref() != Some("1") {
    return;
  }
  if debug {
    eprintln!("wgpu supervisor: nvidia-smi --gpu-reset");
  }
  let _ = Command::new("nvidia-smi")
    .args(["--gpu-reset", "-i", "0"])
    .status();
}

fn snap_paths(dir: &Path) -> (PathBuf, PathBuf, PathBuf, PathBuf, PathBuf, PathBuf) {
  (
    dir.join("node.bin"),
    dir.join("vars.bin"),
    dir.join("rbag.bin"),
    dir.join("work.bin"),
    dir.join("meta.bin"),
    dir.join("itrs.bin"),
  )
}

fn write_u32s(path: &Path, data: &[u32]) -> Result<(), String> {
  let bytes: &[u8] = bytemuck::cast_slice(data);
  std::fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

fn read_snap_u32s(path: &Path) -> Result<Vec<u32>, String> {
  let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
  if bytes.len() % 4 != 0 {
    return Err(format!("{}: not u32-aligned", path.display()));
  }
  Ok(bytemuck::cast_slice(&bytes).to_vec())
}

fn run_inner(book: &hvm::Book) -> Result<bool, String> {
  let main_id = book
    .defs
    .iter()
    .position(|def| def.name == "main")
    .ok_or("missing @main")?;
  let packed = pack_book(book)?;

  fn request_adapter() -> Result<(wgpu::Instance, wgpu::Adapter), String> {
    let want = env::var("HVM_WGPU_BACKEND").unwrap_or_else(|_| "vulkan".into());
    let backends = match want.to_ascii_lowercase().as_str() {
      "dx12" | "d3d12" => wgpu::Backends::DX12,
      "gl" | "gles" => wgpu::Backends::GL,
      _ => wgpu::Backends::VULKAN,
    };
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends;
    let instance = wgpu::Instance::new(desc);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
      power_preference: wgpu::PowerPreference::HighPerformance,
      compatible_surface: None,
      force_fallback_adapter: false,
      apply_limit_buckets: false,
    }))
    .map_err(|e| format!("no GPU adapter ({want}): {e}"))?;
    Ok((instance, adapter))
  }

  let (mut instance, mut adapter) = request_adapter()?;
  let info = adapter.get_info();
  let limits = adapter.limits();
  let heap = heap_len_for(&limits);
  let nthreads = nthreads_len(heap);
  let rspan = match env::var("HVM_WGPU_RSPAN") {
    Ok(s) => s.parse().unwrap_or(DEFAULT_RSPAN),
    Err(_) => DEFAULT_RSPAN,
  };
  let overflow = DEFAULT_OVERFLOW;
  let steps = match env::var("HVM_WGPU_STEPS") {
    Ok(s) => s.parse().unwrap_or(STEPS_PER_DISPATCH),
    Err(_) => STEPS_PER_DISPATCH,
  };
  let rbag_len = nthreads * rspan + overflow;
  let shader_src = shader_for_book(book, nthreads);
  let debug = env::var("HVM_WGPU_DEBUG").ok().as_deref() == Some("1");
  if debug {
    eprintln!(
      "wgpu adapter: {} ({:?}) storage={} workgroup_mem={} int64={} threads={} rspan={} heap={}",
      info.name,
      info.backend,
      limits.max_storage_buffer_binding_size,
      limits.max_compute_workgroup_storage_size,
      adapter.features().contains(wgpu::Features::SHADER_INT64),
      nthreads,
      rspan,
      heap,
    );
    eprintln!("wgpu steps={steps}");
  }

  let node_bytes = heap as u64 * 8;
  let vars_bytes = heap as u64 * 4;
  let rbag_bytes = rbag_len as u64 * 8;
  let book_bytes = (packed.len() * 4) as u64;
  let need = node_bytes.max(vars_bytes).max(rbag_bytes).max(book_bytes);
  if need > limits.max_storage_buffer_binding_size as u64 {
    return Err(format!(
      "adapter max_storage_buffer_binding_size={} < needed {need} (lower HVM_WGPU_HEAP)",
      limits.max_storage_buffer_binding_size
    ));
  }

  let mut req = wgpu::Limits::default();
  req.max_storage_buffer_binding_size = limits.max_storage_buffer_binding_size;
  req.max_buffer_size = limits.max_buffer_size;
  req.max_compute_workgroup_storage_size = limits.max_compute_workgroup_storage_size;

  let mut features = wgpu::Features::empty();
  let have64 = adapter.features().contains(wgpu::Features::SHADER_INT64)
    && adapter.features().contains(wgpu::Features::SHADER_INT64_ATOMIC_ALL_OPS);
  if have64 {
    features |= wgpu::Features::SHADER_INT64 | wgpu::Features::SHADER_INT64_ATOMIC_ALL_OPS;
  } else {
    return Err(
      "this adapter lacks SHADER_INT64_ATOMIC_ALL_OPS; run-wgpu needs 64-bit pair atomics \
       so parallel lanes do not tear CON/DUP ports (which looks like cloning a non-affine REF)"
        .into(),
    );
  }
  let boot_fst = hvm::Port::new(hvm::REF, main_id as u32).0;
  let snap_dir = env::var("HVM_WGPU_SNAP").ok().map(PathBuf::from);
  let resume = snap_dir.as_ref().map(|d| d.join("node.bin").exists()).unwrap_or(false);

  let (mut node_cpu, mut vars_cpu, mut rbag_cpu, mut workers_cpu, mut meta_cpu, mut host_itrs) =
    if resume {
      let d = snap_dir.as_ref().unwrap();
      let (np, vp, rp, wp, mp, ip) = snap_paths(d);
      let itrs = {
        let b = std::fs::read(&ip).map_err(|e| e.to_string())?;
        u64::from_le_bytes(b.try_into().map_err(|_| "itrs.bin")?)
      };
      if debug {
        eprintln!("wgpu resume itrs={itrs} from {}", d.display());
      }
      (
        read_snap_u32s(&np)?,
        read_snap_u32s(&vp)?,
        read_snap_u32s(&rp)?,
        read_snap_u32s(&wp)?,
        {
          let v = read_snap_u32s(&mp)?;
          let mut a = [0u32; META_WORDS];
          a.copy_from_slice(&v[..META_WORDS.min(v.len())]);
          a
        },
        itrs,
      )
    } else {
      let mut node_cpu = vec![0u32; heap as usize * 2];
      let mut vars_cpu = vec![0u32; heap as usize];
      vars_cpu[0] = NONE;
      let mut rbag_cpu = vec![0u32; rbag_len as usize * 2];
      rbag_cpu[0] = boot_fst;
      rbag_cpu[1] = ROOT;
      let mut workers_cpu = vec![0u32; nthreads as usize * W_STRIDE];
      let span = (heap / nthreads).max(1);
      for tid in 0..nthreads as usize {
        let base = (tid as u32 * span).min(heap.saturating_sub(1));
        workers_cpu[tid * W_STRIDE + W_NPUT] = base;
        workers_cpu[tid * W_STRIDE + W_VPUT] = base;
      }
      workers_cpu[W_RLEN] = 1;
      workers_cpu[W_RPUT] = 1;
      let mut meta_cpu = [0u32; META_WORDS];
      meta_cpu[M_NLEN] = heap;
      meta_cpu[M_VLEN] = heap;
      meta_cpu[M_MAX] = steps;
      meta_cpu[M_NTHREADS] = nthreads;
      meta_cpu[M_RSPAN] = rspan;
      meta_cpu[M_NROVER] = 1;
      meta_cpu[M_VROVER] = 1;
      meta_cpu[M_OFLOW] = 0;
      meta_cpu[M_OFLOW_BASE] = nthreads * rspan;
      meta_cpu[M_OFLOW_CAP] = overflow;
      (node_cpu, vars_cpu, rbag_cpu, workers_cpu, meta_cpu, 0u64)
    };
  meta_cpu[M_MAX] = steps;
  meta_cpu[M_NTHREADS] = nthreads;
  meta_cpu[M_NLEN] = heap;
  meta_cpu[M_VLEN] = heap;

  let start = std::time::Instant::now();
  let mut turns = 0u32;
  let mut prev_itrs = u32::MAX;
  let mut stall = 0u32;
  let mut nodes_out: Vec<u32>;
  let mut vars_out: Vec<u32>;

  'sessions: loop {
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
      label: Some("hvm-wgpu"),
      required_features: features,
      required_limits: req.clone(),
      memory_hints: Default::default(),
      trace: Default::default(),
      experimental_features: Default::default(),
    }))
    .map_err(|e| format!("no GPU device: {e}"))?;
    device.on_uncaptured_error(Arc::new(|err| {
      eprintln!("run-wgpu wgpu: {err}");
    }));

    let node_buf = storage_init(&device, bytemuck::cast_slice(&node_cpu));
    let vars_buf = storage_init(&device, bytemuck::cast_slice(&vars_cpu));
    let rbag_buf = storage_init(&device, bytemuck::cast_slice(&rbag_cpu));
    let book_buf = storage_init(&device, bytemuck::cast_slice(&packed));
    let meta_buf = storage_init(&device, bytemuck::cast_slice(&meta_cpu));
    let worker_buf = storage_init(&device, bytemuck::cast_slice(&workers_cpu));

    let shader_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label: Some("hvm.wgsl"),
      source: wgpu::ShaderSource::Wgsl(shader_src.as_str().into()),
    });
    if let Some(err) = pollster::block_on(shader_scope.pop()) {
      return Err(format!("shader: {err}"));
    }
    let pipe_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
      label: Some("hvm-eval"),
      layout: None,
      module: &module,
      entry_point: Some("evaluator"),
      compilation_options: Default::default(),
      cache: None,
    });
    if let Some(err) = pollster::block_on(pipe_scope.pop()) {
      return Err(format!("pipeline: {err}"));
    }
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("hvm-bind"),
      layout: &pipeline.get_bind_group_layout(0),
      entries: &[
        bind_entry(0, &node_buf),
        bind_entry(1, &vars_buf),
        bind_entry(2, &rbag_buf),
        bind_entry(3, &book_buf),
        bind_entry(4, &meta_buf),
        bind_entry(5, &worker_buf),
      ],
    });
    let groups = nthreads.div_ceil(WG);
    // Let WDDM finish the ~400 MiB heap upload before the first long kernel.
    device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("poll upload: {e}"))?;
    std::thread::sleep(std::time::Duration::from_millis(200));
    let worker_words = nthreads as usize * W_STRIDE;
    let meta_staging = device.create_buffer(&wgpu::BufferDescriptor {
      label: Some("hvm-meta-read"),
      size: (META_WORDS * 4) as u64,
      usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
      mapped_at_creation: false,
    });
    let worker_staging = device.create_buffer(&wgpu::BufferDescriptor {
      label: Some("hvm-work-read"),
      size: (worker_words * 4) as u64,
      usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
      mapped_at_creation: false,
    });

    let mut kernels = 0u32;
    let mut last_m = [0u32; META_WORDS];
    last_m.copy_from_slice(&meta_cpu);
    loop {
      let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hvm-dispatch"),
      });
      for _ in 0..KERNELS_PER_SUBMIT {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
          label: Some("hvm-eval"),
          timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
      }
      queue.submit(Some(encoder.finish()));
      kernels = kernels.saturating_add(KERNELS_PER_SUBMIT);
      let poll = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        device.poll(wgpu::PollType::wait_indefinitely())
      }));
      match poll {
        Ok(Ok(_)) => {}
        Ok(Err(e)) => return Err(format!("poll: {e}")),
        Err(_) => {
          eprintln!(
            "wgpu device-lost after {turns} turns / {host_itrs} itrs (exit 101 for supervisor retry)"
          );
          std::process::exit(101);
        }
      }

      turns += 1;
      let check = kernels >= RECYCLE_AFTER_KERNELS || kernels % CHECK_EVERY == 0;
      if !check {
        if turns >= MAX_TURNS {
          return Err(format!("turn cap ({MAX_TURNS}) itrs={host_itrs}"));
        }
        continue;
      }

      let (m, w) = read_meta_and_workers(
        &device,
        &queue,
        &meta_buf,
        &meta_staging,
        &worker_buf,
        &worker_staging,
        worker_words,
      );
      last_m = [0u32; META_WORDS];
      last_m.copy_from_slice(&m[..META_WORDS]);
      if m[M_ERR] != 0 {
        return Err(err_msg(m[M_ERR]));
      }
      if m[M_OOM] != 0 {
        return Err(format!(
          "OOM after {} itrs (heap={heap}, threads={nthreads}, rspan={rspan}). Set HVM_WGPU_HEAP larger.",
          host_itrs + m[M_ITRS] as u64
        ));
      }
      let mut remaining = m[M_OFLOW];
      for tid in 0..nthreads as usize {
        remaining = remaining.saturating_add(w[tid * W_STRIDE + W_RLEN]);
      }
      let itrs_now = m[M_ITRS];
      if remaining == 0 {
        host_itrs += itrs_now as u64;
        vars_out = read_u32s(&device, &queue, &vars_buf, heap as usize);
        nodes_out = read_u32s(&device, &queue, &node_buf, heap as usize * 2);
        break 'sessions;
      }
      if turns > CHECK_EVERY && itrs_now == prev_itrs {
        stall += 1;
        if stall >= 8 {
          return Err(format!(
            "stalled at {} itrs with remain={remaining} (heap={heap}, threads={nthreads})",
            host_itrs + itrs_now as u64
          ));
        }
      } else {
        stall = 0;
      }
      prev_itrs = itrs_now;
      if debug {
        eprintln!(
          "wgpu turn={turns} itrs={} remain={remaining}",
          host_itrs + itrs_now as u64
        );
      }
      if turns >= MAX_TURNS {
        return Err(format!(
          "turn cap ({MAX_TURNS}) with remain={remaining} itrs={}",
          host_itrs + itrs_now as u64
        ));
      }
      if kernels >= RECYCLE_AFTER_KERNELS {
        host_itrs += itrs_now as u64;
        node_cpu = read_u32s(&device, &queue, &node_buf, heap as usize * 2);
        vars_cpu = read_u32s(&device, &queue, &vars_buf, heap as usize);
        rbag_cpu = read_u32s(&device, &queue, &rbag_buf, rbag_len as usize * 2);
        workers_cpu = w;
        meta_cpu = last_m;
        meta_cpu[M_ITRS] = 0;
        meta_cpu[M_RLEN] = remaining;
        if let Some(d) = snap_dir.as_ref() {
          let (np, vp, rp, wp, mp, ip) = snap_paths(d);
          write_u32s(&np, &node_cpu)?;
          write_u32s(&vp, &vars_cpu)?;
          write_u32s(&rp, &rbag_cpu)?;
          write_u32s(&wp, &workers_cpu)?;
          write_u32s(&mp, &meta_cpu)?;
          std::fs::write(&ip, host_itrs.to_le_bytes()).map_err(|e| e.to_string())?;
          if debug {
            eprintln!("wgpu worker recycle itrs={host_itrs}");
          }
          return Ok(false);
        }
        return Err("recycle requested but HVM_WGPU_SNAP unset".into());
      }
    }
  }
  let duration = start.elapsed();
  let itrs = host_itrs;
  let vars = vars_out;
  let nodes = nodes_out;

  let net = hvm::GNet::new(heap as usize, heap as usize);
  for i in 0..heap as usize {
    let lo = remap_port(nodes[i * 2], heap) as u64;
    let hi = remap_port(nodes[i * 2 + 1], heap) as u64;
    net.node_create(i, hvm::Pair(lo | (hi << 32)));
    net.vars_create(i, hvm::Port(remap_port(vars[i], heap)));
  }

  let mut fids = BTreeMap::new();
  for (fid, def) in book.defs.iter().enumerate() {
    fids.insert(fid as hvm::Val, def.name.clone());
  }
  let root = hvm::Port(remap_port(vars[0], heap));
  if root.0 == 0 || root.0 == NONE {
    println!("Readback failed (empty root).");
  } else if let Some(tree) = ast::Tree::readback(&net, net.enter(root), &fids) {
    println!("Result: {}", tree.show());
  } else {
    println!("Readback failed.");
  }

  println!("- ITRS: {}", itrs);
  println!("- LANES: {}", nthreads);
  println!("- TIME: {:.2}s", duration.as_secs_f64());
  println!(
    "- MIPS: {:.2}",
    itrs as f64 / duration.as_secs_f64().max(1e-9) / 1_000_000.0
  );
  Ok(true)
}

const PLAIN_NODE_ACCESS: &str = r#"// NODE_ACCESS_START — replaced by a non-atomic copy when nthreads==1
fn nld(i: u32) -> u64 { return node[i]; }
fn nst(i: u32, v: u64) { node[i] = v; }
fn nxchg(i: u32, v: u64) -> u64 { let t = node[i]; node[i] = v; return t; }
fn ncas0(i: u32) -> bool {
  if node[i] != u64(0) { return false; }
  node[i] = u64(1);
  return true;
}
fn vld(i: u32) -> u32 { return vars[i]; }
fn vst(i: u32, v: u32) { vars[i] = v; }
fn vxchg(i: u32, v: u32) -> u32 { let t = vars[i]; vars[i] = v; return t; }
fn vcas(i: u32, exp: u32, v: u32) -> bool {
  if vars[i] != exp { return false; }
  vars[i] = v;
  return true;
}
// NODE_ACCESS_END"#;

fn shader_for_book(book: &hvm::Book, nthreads: u32) -> String {
  let start = SHADER
    .find("fn interact_call(a: u32, b: u32)")
    .expect("interact_call marker");
  let end = SHADER.find("fn interact_one(").expect("interact_one marker");
  let mut out = String::with_capacity(SHADER.len() + 4096);
  out.push_str(&SHADER[..start]);
  out.push_str(&compile_interact_wgsl(book));
  out.push('\n');
  out.push_str(&SHADER[end..]);
  if nthreads <= 1 {
    let acc_start = out.find("// NODE_ACCESS_START").expect("access start");
    let acc_end = out.find("// NODE_ACCESS_END").expect("access end") + "// NODE_ACCESS_END".len();
    out.replace_range(acc_start..acc_end, PLAIN_NODE_ACCESS);
    out = out.replace(
      "var<storage, read_write> node: array<atomic<u64>>;",
      "var<storage, read_write> node: array<u64>;",
    );
    out = out.replace(
      "var<storage, read_write> vars: array<atomic<u32>>;",
      "var<storage, read_write> vars: array<u32>;",
    );
  }
  out
}

fn compile_interact_wgsl(book: &hvm::Book) -> String {
  let mut out = String::new();
  for (fid, def) in book.defs.iter().enumerate() {
    out.push_str(&format!("fn ic_{fid}(a: u32, b: u32) -> bool {{\n"));
    if def.safe {
      out.push_str("  if tag_of(b) == DUP { return interact_eras(a, b); }\n");
    }
    out.push_str(&format!(
      "  if !get_resources({}u, {}u) {{ return false; }}\n",
      def.node.len(),
      def.vars
    ));
    for i in 0..def.vars {
      out.push_str(&format!("  vst(vloc[{i}u], NONE);\n"));
    }
    for (i, p) in def.node.iter().enumerate() {
      let lo = (p.0 & 0xFFFF_FFFF) as u32;
      let hi = (p.0 >> 32) as u32;
      out.push_str(&format!(
        "  node_store(nloc[{i}u], vec2<u32>(adjust_port(0x{lo:08X}u), adjust_port(0x{hi:08X}u)));\n"
      ));
    }
    for p in &def.rbag {
      let lo = (p.0 & 0xFFFF_FFFF) as u32;
      let hi = (p.0 >> 32) as u32;
      out.push_str(&format!(
        "  link(adjust_port(0x{lo:08X}u), adjust_port(0x{hi:08X}u));\n"
      ));
    }
    out.push_str(&format!(
      "  link(adjust_port(0x{:08X}u), b);\n  return true;\n}}\n\n",
      def.root.0
    ));
  }
  out.push_str("fn interact_call(a: u32, b: u32) -> bool {\n");
  // Touch `book` so naga keeps @binding(3); compiled defs no longer load it.
  out.push_str("  if book[0] == 0u { atomicStore(&ctl[M_ERR], 1u); return false; }\n");
  out.push_str("  let fid = val_of(a) & 0x0FFFFFFFu;\n");
  out.push_str("  switch fid {\n");
  for fid in 0..book.defs.len() {
    out.push_str(&format!("    case {fid}u: {{ return ic_{fid}(a, b); }}\n"));
  }
  out.push_str("    default: { atomicStore(&ctl[M_ERR], 1u); return false; }\n");
  out.push_str("  }\n}\n");
  out
}

fn pack_book(book: &hvm::Book) -> Result<Vec<u32>, String> {
  let ndefs = book.defs.len() as u32;
  if ndefs == 0 {
    return Err("empty book".into());
  }
  let header_words = 1 + ndefs * 8;
  let mut pairs = Vec::new();
  let mut headers = Vec::with_capacity(book.defs.len());
  for def in &book.defs {
    if def.node.len() > MAX_DEF_SLOTS || def.vars > MAX_DEF_SLOTS {
      return Err(format!(
        "def '{}' too large for wgpu v1 (nodes={}, vars={}, max={MAX_DEF_SLOTS})",
        def.name,
        def.node.len(),
        def.vars
      ));
    }
    let rbag_off = header_words + pairs.len() as u32;
    for p in &def.rbag {
      pairs.push((p.0 & 0xFFFF_FFFF) as u32);
      pairs.push((p.0 >> 32) as u32);
    }
    let node_off = header_words + pairs.len() as u32;
    for p in &def.node {
      pairs.push((p.0 & 0xFFFF_FFFF) as u32);
      pairs.push((p.0 >> 32) as u32);
    }
    headers.push([
      u32::from(def.safe),
      def.rbag.len() as u32,
      def.node.len() as u32,
      def.vars as u32,
      def.root.0,
      rbag_off,
      node_off,
      0,
    ]);
  }
  let mut out = Vec::with_capacity(header_words as usize + pairs.len());
  out.push(ndefs);
  for h in headers {
    out.extend_from_slice(&h);
  }
  out.extend_from_slice(&pairs);
  if out.len() < 4 {
    out.resize(4, 0);
  }
  Ok(out)
}

fn remap_port(p: u32, heap: u32) -> u32 {
  // GPU treats ROOT as vars[0]; host GNet is only `heap` long.
  if p == ROOT {
    return 0;
  }
  if (p & 7) == 0 && (p >> 3) >= heap {
    return 0;
  }
  p
}

fn heap_len_for(limits: &wgpu::Limits) -> u32 {
  let max_nodes = ((limits.max_storage_buffer_binding_size as u64) / 8).min(u32::MAX as u64) as u32;
  let want = match env::var("HVM_WGPU_HEAP") {
    Ok(s) => s.parse().unwrap_or(DEFAULT_HEAP),
    Err(_) => DEFAULT_HEAP,
  };
  want.min(max_nodes).max(16).next_power_of_two()
}

fn nthreads_len(heap: u32) -> u32 {
  let want = match env::var("HVM_WGPU_THREADS") {
    Ok(s) => s.parse().unwrap_or(DEFAULT_NTHREADS),
    Err(_) => DEFAULT_NTHREADS,
  };
  // Extra lanes in the 64-wide group return immediately when nthreads < WG.
  let cap = (heap / 16).max(1);
  want.clamp(1, cap)
}

fn storage_init(device: &wgpu::Device, data: &[u8]) -> wgpu::Buffer {
  device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: None,
    contents: data,
    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
  })
}

fn bind_entry<'a>(binding: u32, buf: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
  wgpu::BindGroupEntry {
    binding,
    resource: buf.as_entire_binding(),
  }
}

fn read_meta_and_workers(
  device: &wgpu::Device,
  queue: &wgpu::Queue,
  meta_src: &wgpu::Buffer,
  meta_staging: &wgpu::Buffer,
  work_src: &wgpu::Buffer,
  work_staging: &wgpu::Buffer,
  work_count: usize,
) -> ([u32; META_WORDS], Vec<u32>) {
  let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
  encoder.copy_buffer_to_buffer(meta_src, 0, meta_staging, 0, (META_WORDS * 4) as u64);
  encoder.copy_buffer_to_buffer(work_src, 0, work_staging, 0, (work_count * 4) as u64);
  queue.submit(Some(encoder.finish()));
  let meta_slice = meta_staging.slice(..);
  let work_slice = work_staging.slice(..);
  meta_slice.map_async(wgpu::MapMode::Read, |_| ());
  work_slice.map_async(wgpu::MapMode::Read, |_| ());
  device.poll(wgpu::PollType::wait_indefinitely()).expect("poll status");
  let meta_mapped = meta_slice.get_mapped_range().expect("map meta");
  let work_mapped = work_slice.get_mapped_range().expect("map work");
  let mut meta = [0u32; META_WORDS];
  meta.copy_from_slice(&bytemuck::cast_slice(&meta_mapped)[..META_WORDS]);
  let work = bytemuck::cast_slice(&work_mapped).to_vec();
  drop(meta_mapped);
  drop(work_mapped);
  meta_staging.unmap();
  work_staging.unmap();
  (meta, work)
}

fn read_u32s(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, count: usize) -> Vec<u32> {
  let bytes = (count * 4) as u64;
  let staging = device.create_buffer(&wgpu::BufferDescriptor {
    label: None,
    size: bytes,
    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
  });
  read_u32s_into(device, queue, src, &staging, count)
}

fn read_u32s_into(
  device: &wgpu::Device,
  queue: &wgpu::Queue,
  src: &wgpu::Buffer,
  staging: &wgpu::Buffer,
  count: usize,
) -> Vec<u32> {
  let bytes = (count * 4) as u64;
  let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
  encoder.copy_buffer_to_buffer(src, 0, staging, 0, bytes);
  queue.submit(Some(encoder.finish()));
  let slice = staging.slice(..);
  slice.map_async(wgpu::MapMode::Read, |_| ());
  device.poll(wgpu::PollType::wait_indefinitely()).expect("poll map");
  let mapped = slice.get_mapped_range().expect("map");
  let out = bytemuck::cast_slice(&mapped).to_vec();
  drop(mapped);
  staging.unmap();
  out
}

fn err_msg(code: u32) -> String {
  match code {
    1 => "invalid fid in CALL".into(),
    2 => "attempt to clone a non-affine global reference".into(),
    3 => "def too large for nloc/vloc".into(),
    _ => format!("device error {code}"),
  }
}
