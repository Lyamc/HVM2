//! WebGPU interpreter for existing HVM books.
//!
//! Same `ast::Book` / `.hvm` encoding as `hvm run`. Bend programs are not
//! rewritten: `ROOT` (`0xFFFFFFF8`) is remapped to `vars[0]` only inside this
//! backend so a 2^16 heap can host it. Readback uses `ast::Tree::readback`.

use crate::ast;
use crate::hvm;
use std::collections::BTreeMap;
use std::env;
use std::sync::Arc;
use wgpu::util::DeviceExt;

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
const W_STRIDE: usize = 8;

const MAX_DEF_SLOTS: usize = 1024;
const DEFAULT_HEAP: u32 = 1 << 24; // 16M nodes (~128 MiB); adapter on this box allows 2 GiB
const WG: u32 = 64;
const DEFAULT_NTHREADS: u32 = 64 * 64; // 4096; CUDA is 128*128 = 16384
const DEFAULT_RSPAN: u32 = 512;
const DEFAULT_OVERFLOW: u32 = 1 << 20; // global redex overflow (CUDA-style shared bag)
const MAX_TURNS: u32 = 1 << 16;
const STEPS_PER_DISPATCH: u32 = 256; // keep each dispatch under the Windows TDR (~2s)

pub fn run(book: &hvm::Book) {
  match run_inner(book) {
    Ok(()) => {}
    Err(e) => {
      eprintln!("run-wgpu: {e}");
      std::process::exit(1);
    }
  }
}

fn run_inner(book: &hvm::Book) -> Result<(), String> {
  let main_id = book
    .defs
    .iter()
    .position(|def| def.name == "main")
    .ok_or("missing @main")?;
  let packed = pack_book(book)?;

  let instance = wgpu::Instance::default();
  let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
    power_preference: wgpu::PowerPreference::HighPerformance,
    compatible_surface: None,
    force_fallback_adapter: false,
    apply_limit_buckets: false,
  }))
  .map_err(|e| format!("no GPU adapter: {e}"))?;

  let info = adapter.get_info();
  let limits = adapter.limits();
  let heap = heap_len_for(&limits);
  let nthreads = nthreads_len(heap);
  let rspan = DEFAULT_RSPAN;
  let overflow = DEFAULT_OVERFLOW;
  let rbag_len = nthreads * rspan + overflow;
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
      heap
    );
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
  let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
    label: Some("hvm-wgpu"),
    required_features: features,
    required_limits: req,
    memory_hints: Default::default(),
    trace: Default::default(),
    experimental_features: Default::default(),
  }))
  .map_err(|e| format!("no GPU device: {e}"))?;

  let boot_fst = hvm::Port::new(hvm::REF, main_id as u32).0;
  let node_init = vec![0u32; heap as usize * 2];
  let mut vars_init = vec![0u32; heap as usize];
  vars_init[0] = NONE;
  let mut rbag_init = vec![0u32; rbag_len as usize * 2];
  rbag_init[0] = boot_fst;
  rbag_init[1] = ROOT;

  let mut workers = vec![0u32; nthreads as usize * W_STRIDE];
  let span = (heap / nthreads).max(1);
  for tid in 0..nthreads as usize {
    let base = (tid as u32 * span).min(heap.saturating_sub(1));
    workers[tid * W_STRIDE + W_NPUT] = base;
    workers[tid * W_STRIDE + W_VPUT] = base;
  }
  workers[W_RLEN] = 1;

  let mut meta = [0u32; META_WORDS];
  meta[M_NLEN] = heap;
  meta[M_VLEN] = heap;
  meta[M_MAX] = STEPS_PER_DISPATCH;
  meta[M_NTHREADS] = nthreads;
  meta[M_RSPAN] = rspan;
  meta[M_NROVER] = 1;
  meta[M_VROVER] = 1;
  meta[M_OFLOW] = 0;
  meta[M_OFLOW_BASE] = nthreads * rspan;
  meta[M_OFLOW_CAP] = overflow;

  device.on_uncaptured_error(Arc::new(|err| {
    eprintln!("run-wgpu wgpu: {err}");
  }));

  let node_buf = storage_init(&device, bytemuck::cast_slice(&node_init));
  let vars_buf = storage_init(&device, bytemuck::cast_slice(&vars_init));
  let rbag_buf = storage_init(&device, bytemuck::cast_slice(&rbag_init));
  let book_buf = storage_init(&device, bytemuck::cast_slice(&packed));
  let meta_buf = storage_init(&device, bytemuck::cast_slice(&meta));
  let worker_buf = storage_init(&device, bytemuck::cast_slice(&workers));

  let shader_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
  let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: Some("hvm.wgsl"),
    source: wgpu::ShaderSource::Wgsl(SHADER.into()),
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

  let start = std::time::Instant::now();
  let mut turns = 0u32;
  let mut prev_itrs = u32::MAX;
  loop {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
      label: Some("hvm-dispatch"),
    });
    {
      let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("hvm-eval"),
        timestamp_writes: None,
      });
      pass.set_pipeline(&pipeline);
      pass.set_bind_group(0, &bind, &[]);
      pass.dispatch_workgroups(groups, 1, 1);
    }
    queue.submit(Some(encoder.finish()));
    device
      .poll(wgpu::PollType::wait_indefinitely())
      .map_err(|e| format!("poll: {e}"))?;

    let m = read_u32s(&device, &queue, &meta_buf, META_WORDS);
    if m[M_ERR] != 0 {
      return Err(err_msg(m[M_ERR]));
    }
    if m[M_OOM] != 0 {
      return Err(format!(
        "OOM after {} itrs (heap={heap}, threads={nthreads}, rspan={rspan}). Set HVM_WGPU_HEAP larger.",
        m[M_ITRS]
      ));
    }
    let remaining = m[M_RLEN].saturating_add(m[M_OFLOW]);
    let itrs_now = m[M_ITRS];
    // Zero the remaining-work accumulator for the next pass.
    let zero = 0u32;
    queue.write_buffer(&meta_buf, (M_RLEN * 4) as u64, bytemuck::bytes_of(&zero));
    // rlen can stay stale; if a full dispatch produced no new interacts, we are done.
    if remaining == 0 || (turns > 0 && itrs_now == prev_itrs) {
      break;
    }
    prev_itrs = itrs_now;
    turns += 1;
    if turns >= MAX_TURNS {
      return Err(format!(
        "turn cap ({MAX_TURNS}) with rlen={} itrs={}",
        m[M_RLEN], m[M_ITRS]
      ));
    }
  }
  let duration = start.elapsed();

  let m = read_u32s(&device, &queue, &meta_buf, META_WORDS);
  let itrs = m[M_ITRS] as u64;
  let vars = read_u32s(&device, &queue, &vars_buf, heap as usize);
  let nodes = read_u32s(&device, &queue, &node_buf, heap as usize * 2);

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
  Ok(())
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
  // Each worker needs a few node/var slots; keep a multiple of the workgroup size.
  let cap = (heap / 16).max(WG);
  want.clamp(WG, cap) / WG * WG
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

fn read_u32s(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, count: usize) -> Vec<u32> {
  let bytes = (count * 4) as u64;
  let staging = device.create_buffer(&wgpu::BufferDescriptor {
    label: None,
    size: bytes,
    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
  });
  let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
  encoder.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
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
