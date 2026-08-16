// Multi-lane HVM2 evaluator (C/Rust steal-bag shape). Semantics match src/hvm.rs.
// ROOT (0xFFFFFFF8) is vars[0]; books are the same as the CPU interpreter.

@group(0) @binding(0) var<storage, read_write> node: array<atomic<u64>>;
@group(0) @binding(1) var<storage, read_write> vars: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> rbag: array<atomic<u64>>;
@group(0) @binding(3) var<storage, read> book: array<u32>;
@group(0) @binding(4) var<storage, read_write> ctl: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> worker: array<atomic<u32>>;

const VAR: u32 = 0u;
const REF: u32 = 1u;
const ERA: u32 = 2u;
const NUM: u32 = 3u;
const CON: u32 = 4u;
const DUP: u32 = 5u;
const OPR: u32 = 6u;
const SWI: u32 = 7u;

const LINK: u32 = 0u;
const CALL: u32 = 1u;
const VOID: u32 = 2u;
const ERAS: u32 = 3u;
const ANNI: u32 = 4u;
const COMM: u32 = 5u;
const OPER: u32 = 6u;
const SWIT: u32 = 7u;

const TY_SYM: u32 = 0u;
const TY_U24: u32 = 1u;
const TY_I24: u32 = 2u;
const TY_F24: u32 = 3u;
const OP_ADD: u32 = 4u;
const OP_SUB: u32 = 5u;
const FP_SUB: u32 = 6u;
const OP_MUL: u32 = 7u;
const OP_DIV: u32 = 8u;
const FP_DIV: u32 = 9u;
const OP_REM: u32 = 10u;
const FP_REM: u32 = 11u;
const OP_EQ: u32  = 12u;
const OP_NEQ: u32 = 13u;
const OP_LT: u32  = 14u;
const OP_GT: u32  = 15u;
const OP_AND: u32 = 16u;
const OP_OR: u32  = 17u;
const OP_XOR: u32 = 18u;
const OP_SHL: u32 = 19u;
const FP_SHL: u32 = 20u;
const OP_SHR: u32 = 21u;
const FP_SHR: u32 = 22u;

const ROOT: u32 = 0xFFFFFFF8u;
const NONE: u32 = 0xFFFFFFFFu;
const U24_MAX: u32 = 0x00FFFFFFu;
const I24_MAX: i32 = 8388607;
const I24_MIN: i32 = -8388608;
const MAX_SLOTS: u32 = 1024u;

const M_ITRS: u32 = 0u;
const M_OOM: u32 = 1u;
const M_ERR: u32 = 2u;
const M_NLEN: u32 = 3u;
const M_VLEN: u32 = 4u;
const M_MAX: u32 = 5u;
const M_NTHREADS: u32 = 6u;
const M_RSPAN: u32 = 7u;
const M_RLEN: u32 = 8u;
const M_NROVER: u32 = 9u;
const M_VROVER: u32 = 10u;
const M_OFLOW: u32 = 11u;
const M_OFLOW_BASE: u32 = 12u;
const M_OFLOW_CAP: u32 = 13u;

const W_NPUT: u32 = 0u;
const W_VPUT: u32 = 1u;
const W_NWRAP: u32 = 2u;
const W_VWRAP: u32 = 3u;
const W_RLEN: u32 = 4u;
const W_SIDX: u32 = 5u;
const W_STRIDE: u32 = 8u;

// VAR  REF  ERA  NUM  CON  DUP  OPR  SWI
const RULE: array<u32, 64> = array<u32, 64>(
  0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
  0u, 2u, 2u, 2u, 1u, 1u, 1u, 1u,
  0u, 2u, 2u, 2u, 3u, 3u, 3u, 3u,
  0u, 2u, 2u, 2u, 3u, 3u, 6u, 7u,
  0u, 1u, 3u, 3u, 4u, 5u, 5u, 5u,
  0u, 1u, 3u, 3u, 5u, 4u, 5u, 5u,
  0u, 1u, 3u, 6u, 5u, 5u, 4u, 5u,
  0u, 1u, 3u, 7u, 5u, 5u, 5u, 4u,
);

const LMAX: u32 = 64u;

var<private> nloc: array<u32, 1024>;
var<private> vloc: array<u32, 1024>;
var<private> nput: u32;
var<private> vput: u32;
var<private> nwrap: u32;
var<private> vwrap: u32;
var<private> g_tid: u32;
var<private> llo: array<u32, 64>;
var<private> lhi: array<u32, 64>;
var<private> ltop: u32;
var<private> litrs: u32;

fn is_high(rule: u32) -> bool {
  // LINK, VOID, ERAS, ANNI — same mask as hvm.rs is_high_priority
  return ((0x1Du >> rule) & 1u) != 0u;
}

fn tag_of(p: u32) -> u32 { return p & 7u; }
fn val_of(p: u32) -> u32 { return p >> 3u; }
fn mk_port(tag: u32, val: u32) -> u32 { return (val << 3u) | tag; }

fn get_rule(a: u32, b: u32) -> u32 {
  return RULE[tag_of(a) * 8u + tag_of(b)];
}

fn var_idx(p: u32) -> u32 {
  if p == ROOT { return 0u; }
  return val_of(p);
}

fn pack2(a: u32, b: u32) -> u64 {
  return u64(a) | (u64(b) << 32u);
}

fn unpack2(x: u64) -> vec2<u32> {
  return vec2<u32>(u32(x), u32(x >> 32u));
}

fn node_empty(i: u32) -> bool {
  return atomicLoad(&node[i]) == u64(0);
}

fn node_load(i: u32) -> vec2<u32> {
  return unpack2(atomicLoad(&node[i]));
}

fn node_store(i: u32, p: vec2<u32>) {
  atomicStore(&node[i], pack2(p.x, p.y));
}

fn node_take(i: u32) -> vec2<u32> {
  return unpack2(atomicExchange(&node[i], u64(0)));
}

fn wf(tid: u32, field: u32) -> u32 {
  return tid * W_STRIDE + field;
}

fn pop_from(wid: u32, limit: u32) -> vec3<u32> {
  let rspan = atomicLoad(&ctl[M_RSPAN]);
  let base = wid * rspan;
  let start = atomicAdd(&worker[wf(g_tid, W_SIDX)], 1u);
  let n = min(limit, rspan);
  for (var k = 0u; k < n; k++) {
    let idx = base + ((start + k) % rspan);
    let raw = atomicLoad(&rbag[idx]);
    if raw == u64(0) {
      continue;
    }
    let got = atomicExchange(&rbag[idx], u64(0));
    if got == u64(0) {
      continue;
    }
    let lo = u32(got);
    let hi = u32(got >> 32u);
    let cur = atomicLoad(&worker[wf(wid, W_RLEN)]);
    if cur > 0u {
      atomicSub(&worker[wf(wid, W_RLEN)], 1u);
    }
    return vec3<u32>(lo, hi, 1u);
  }
  return vec3<u32>(0u, 0u, 0u);
}

fn try_emit_on(wid: u32, a: u32, b: u32) -> bool {
  let rspan = atomicLoad(&ctl[M_RSPAN]);
  let base = wid * rspan;
  var put = atomicLoad(&worker[wf(wid, W_SIDX)]);
  for (var k = 0u; k < rspan; k++) {
    let i = (put + k) % rspan;
    let idx = base + i;
    let old = atomicCompareExchangeWeak(&rbag[idx], u64(0), pack2(a, b));
    if old.exchanged {
      atomicStore(&worker[wf(wid, W_SIDX)], (i + 1u) % rspan);
      atomicAdd(&worker[wf(wid, W_RLEN)], 1u);
      return true;
    }
  }
  return false;
}

fn emit_global(a: u32, b: u32) {
  if try_emit_on(g_tid, a, b) { return; }
  let n = atomicLoad(&ctl[M_NTHREADS]);
  for (var k = 1u; k < 8u; k++) {
    if k >= n { break; }
    let wid = (g_tid + k) % n;
    if try_emit_on(wid, a, b) { return; }
  }
  let cap = atomicLoad(&ctl[M_OFLOW_CAP]);
  if cap > 0u {
    let i = atomicAdd(&ctl[M_OFLOW], 1u);
    if i < cap {
      let idx = atomicLoad(&ctl[M_OFLOW_BASE]) + i;
      atomicStore(&rbag[idx], pack2(a, b));
      return;
    }
  }
  atomicStore(&ctl[M_OOM], 1u);
}

fn emit(a: u32, b: u32) {
  let rule = get_rule(a, b);
  // High-pri stays local (like Rust/C). Low-pri goes to the steal bag so other lanes can help.
  if (is_high(rule) || ltop < 32u) && ltop < LMAX {
    llo[ltop] = a;
    lhi[ltop] = b;
    ltop = ltop + 1u;
    return;
  }
  emit_global(a, b);
}

fn flush_local() {
  loop {
    if ltop == 0u { break; }
    ltop = ltop - 1u;
    emit_global(llo[ltop], lhi[ltop]);
  }
}

fn pop_redex() -> vec3<u32> {
  if ltop > 0u {
    ltop = ltop - 1u;
    return vec3<u32>(llo[ltop], lhi[ltop], 1u);
  }
  let rspan = atomicLoad(&ctl[M_RSPAN]);
  let own = pop_from(g_tid, rspan);
  if own.z != 0u {
    return own;
  }
  let n = atomicLoad(&ctl[M_NTHREADS]);
  for (var k = 1u; k < 8u; k++) {
    if k >= n { break; }
    let sid = (g_tid + n - k) % n;
    let st = pop_from(sid, 32u);
    if st.z != 0u {
      return st;
    }
  }
  let cap = atomicLoad(&ctl[M_OFLOW_CAP]);
  if cap > 0u {
    let n = atomicLoad(&ctl[M_OFLOW]);
    if n > 0u {
      let r = atomicCompareExchangeWeak(&ctl[M_OFLOW], n, n - 1u);
      if r.exchanged {
        let idx = atomicLoad(&ctl[M_OFLOW_BASE]) + (n - 1u);
        let got = atomicExchange(&rbag[idx], u64(0));
        if got != u64(0) {
          return vec3<u32>(u32(got), u32(got >> 32u), 1u);
        }
      }
    }
  }
  return vec3<u32>(0u, 0u, 0u);
}

fn enter(start: u32) -> u32 {
  var p = start;
  for (var g = 0u; g < 1024u; g++) {
    if tag_of(p) != VAR { return p; }
    let idx = var_idx(p);
    if idx >= atomicLoad(&ctl[M_VLEN]) { return p; }
    let val = atomicExchange(&vars[idx], NONE);
    if val == NONE || val == 0u {
      return p;
    }
    atomicStore(&vars[idx], 0u);
    p = val;
  }
  return p;
}

fn link(aa: u32, bb: u32) {
  var a = aa;
  var b = bb;
  for (var g = 0u; g < 1024u; g++) {
    if tag_of(a) != VAR && tag_of(b) == VAR {
      let x = a; a = b; b = x;
    }
    if tag_of(a) != VAR {
      emit(a, b);
      return;
    }
    b = enter(b);
    let idx = var_idx(a);
    if idx >= atomicLoad(&ctl[M_VLEN]) {
      atomicStore(&ctl[M_OOM], 1u);
      return;
    }
    let a_ = atomicExchange(&vars[idx], b);
    if a_ == NONE {
      return;
    }
    atomicStore(&vars[idx], 0u);
    a = a_;
  }
}

fn part_range(len: u32) -> vec2<u32> {
  let n = atomicLoad(&ctl[M_NTHREADS]);
  let span = max(len / n, 1u);
  let base = g_tid * span;
  if base >= len {
    return vec2<u32>(1u, len);
  }
  var end = base + span;
  if g_tid + 1u == n || end > len {
    end = len;
  }
  return vec2<u32>(base, end);
}

fn rover_nodes(need: u32, already: u32) -> bool {
  let nlen = atomicLoad(&ctl[M_NLEN]);
  var got = already;
  let tries = min(nlen, 8192u);
  for (var t = 0u; t < tries; t++) {
    var slot = atomicAdd(&ctl[M_NROVER], 1u) % nlen;
    if slot == 0u { slot = 1u; }
    // Claim exclusively. 1u is not a valid pair (fst tag VAR val 0).
    let claim = atomicCompareExchangeWeak(&node[slot], u64(0), u64(1));
    if !claim.exchanged { continue; }
    nloc[got] = slot;
    got = got + 1u;
    if got >= need { return true; }
  }
  return false;
}

fn node_alloc(num: u32) -> bool {
  if num == 0u { return true; }
  if num > MAX_SLOTS { return false; }
  let nlen = atomicLoad(&ctl[M_NLEN]);
  let rg = part_range(nlen);
  let base = rg.x;
  let end = rg.y;
  let limit = max(end - base, 1u);
  var got = 0u;
  for (var k = 0u; k < limit; k++) {
    nput = nput + 1u;
    if nput >= end || nput == 0u {
      nput = max(base, 1u);
      nwrap = 1u;
    }
    let claim = atomicCompareExchangeWeak(&node[nput], u64(0), u64(1));
    if !claim.exchanged {
      continue;
    }
    nloc[got] = nput;
    got = got + 1u;
    if got >= num { return true; }
  }
  return rover_nodes(num, got);
}

fn vars_alloc(num: u32) -> bool {
  if num == 0u { return true; }
  if num > MAX_SLOTS { return false; }
  let vlen = atomicLoad(&ctl[M_VLEN]);
  let rg = part_range(vlen);
  let base = rg.x;
  let end = rg.y;
  let limit = max(end - base, 1u);
  var got = 0u;
  for (var k = 0u; k < limit; k++) {
    vput = vput + 1u;
    if vput >= vlen || vput == 0u || vput >= end {
      vput = max(base, 1u);
      vwrap = 1u;
    }
    let claim = atomicCompareExchangeWeak(&vars[vput], 0u, NONE);
    if !claim.exchanged {
      continue;
    }
    vloc[got] = vput;
    got = got + 1u;
    if got >= num { return true; }
  }
  return rover_vars(num, got);
}

fn rover_vars(need: u32, already: u32) -> bool {
  let vlen = atomicLoad(&ctl[M_VLEN]);
  var got = already;
  let tries = min(vlen, 8192u);
  for (var t = 0u; t < tries; t++) {
    var slot = atomicAdd(&ctl[M_VROVER], 1u) % vlen;
    if slot == 0u { slot = 1u; }
    let claim = atomicCompareExchangeWeak(&vars[slot], 0u, NONE);
    if !claim.exchanged { continue; }
    vloc[got] = slot;
    got = got + 1u;
    if got >= need { return true; }
  }
  return false;
}

fn get_resources(need_node: u32, need_vars: u32) -> bool {
  return node_alloc(need_node) && vars_alloc(need_vars);
}

fn adjust_port(p: u32) -> u32 {
  let t = tag_of(p);
  let v = val_of(p);
  if t >= CON {
    if v >= MAX_SLOTS { return p; }
    return mk_port(t, nloc[v]);
  }
  if t == VAR {
    if v >= MAX_SLOTS { return p; }
    return mk_port(t, vloc[v]);
  }
  return p;
}

fn interact_link(a: u32, b: u32) -> bool {
  if !get_resources(0u, 0u) { return false; }
  link(a, b);
  return true;
}

fn interact_void() -> bool {
  return true;
}

fn interact_eras(a: u32, b: u32) -> bool {
  if !get_resources(0u, 0u) { return false; }
  let bi = val_of(b);
  if bi >= atomicLoad(&ctl[M_NLEN]) || node_empty(bi) { return false; }
  let bp = node_take(bi);
  link(a, bp.x);
  link(a, bp.y);
  return true;
}

fn interact_anni(a: u32, b: u32) -> bool {
  if !get_resources(0u, 0u) { return false; }
  let ai = val_of(a);
  let bi = val_of(b);
  let nlen = atomicLoad(&ctl[M_NLEN]);
  if ai >= nlen || bi >= nlen || node_empty(ai) || node_empty(bi) { return false; }
  let ap = node_take(ai);
  let bp = node_take(bi);
  link(ap.x, bp.x);
  link(ap.y, bp.y);
  return true;
}

fn interact_comm(a: u32, b: u32) -> bool {
  if !get_resources(4u, 4u) { return false; }
  let ai = val_of(a);
  let bi = val_of(b);
  let nlen = atomicLoad(&ctl[M_NLEN]);
  if ai >= nlen || bi >= nlen || node_empty(ai) || node_empty(bi) { return false; }
  let ap = node_take(ai);
  let bp = node_take(bi);
  atomicStore(&vars[vloc[0]], NONE);
  atomicStore(&vars[vloc[1]], NONE);
  atomicStore(&vars[vloc[2]], NONE);
  atomicStore(&vars[vloc[3]], NONE);
  node_store(nloc[0], vec2<u32>(mk_port(VAR, vloc[0]), mk_port(VAR, vloc[1])));
  node_store(nloc[1], vec2<u32>(mk_port(VAR, vloc[2]), mk_port(VAR, vloc[3])));
  node_store(nloc[2], vec2<u32>(mk_port(VAR, vloc[0]), mk_port(VAR, vloc[2])));
  node_store(nloc[3], vec2<u32>(mk_port(VAR, vloc[1]), mk_port(VAR, vloc[3])));
  let bt = tag_of(b);
  let at = tag_of(a);
  link(mk_port(bt, nloc[0]), ap.x);
  link(mk_port(bt, nloc[1]), ap.y);
  link(mk_port(at, nloc[2]), bp.x);
  link(mk_port(at, nloc[3]), bp.y);
  return true;
}

fn new_u24(v: u32) -> u32 { return (v << 5u) | TY_U24; }
fn get_u24(n: u32) -> u32 { return n >> 5u; }
fn new_i24(v: i32) -> u32 { return (bitcast<u32>(v) << 5u) | TY_I24; }
fn get_i24(n: u32) -> i32 { return bitcast<i32>(n) << 3 >> 8; }
fn get_typ(n: u32) -> u32 { return n & 31u; }
fn get_sym(n: u32) -> u32 { return n >> 5u; }
fn new_sym(v: u32) -> u32 { return (v << 5u) | TY_SYM; }

fn get_f24(n: u32) -> f32 {
  return bitcast<f32>((n << 3u) & 0xFFFFFF00u);
}

fn new_f24(val: f32) -> u32 {
  let bits = bitcast<u32>(val);
  var shifted = bits >> 8u;
  let lost = bits & 0xFFu;
  let nan = val != val;
  if !nan {
    let t = (lost - ((lost >> 7u) & ~shifted)) >> 7u;
    shifted = shifted + t;
  } else {
    shifted = shifted | 1u;
  }
  return (shifted << 5u) | TY_F24;
}

fn is_num(n: u32) -> bool {
  let t = get_typ(n);
  return t >= TY_U24 && t <= TY_F24;
}

fn is_cast(n: u32) -> bool {
  if get_typ(n) != TY_SYM { return false; }
  let s = get_sym(n);
  return s >= TY_U24 && s <= TY_F24;
}

fn f_as_u32(v: f32) -> u32 {
  if v != v { return 0u; }
  if v <= 0.0 { return 0u; }
  if v >= 4294967296.0 { return 0xFFFFFFFFu; }
  return u32(v);
}

fn f_as_i32(v: f32) -> i32 {
  if v != v { return 0; }
  if v >= 2147483647.0 { return 2147483647; }
  if v <= -2147483648.0 { return -2147483648; }
  return i32(v);
}

fn clamp_u24(v: u32) -> u32 {
  if v > U24_MAX { return U24_MAX; }
  return v;
}

fn clamp_i24(v: i32) -> i32 {
  if v > I24_MAX { return I24_MAX; }
  if v < I24_MIN { return I24_MIN; }
  return v;
}

fn numb_cast(a: u32, b: u32) -> u32 {
  let s = get_sym(a);
  let t = get_typ(b);
  if s == TY_U24 && t == TY_U24 { return b; }
  if s == TY_U24 && t == TY_I24 { return new_u24(bitcast<u32>(get_i24(b))); }
  if s == TY_U24 && t == TY_F24 { return new_u24(clamp_u24(f_as_u32(get_f24(b)))); }
  if s == TY_I24 && t == TY_U24 { return new_i24(bitcast<i32>(get_u24(b))); }
  if s == TY_I24 && t == TY_I24 { return b; }
  if s == TY_I24 && t == TY_F24 { return new_i24(clamp_i24(f_as_i32(get_f24(b)))); }
  if s == TY_F24 && t == TY_U24 { return new_f24(f32(get_u24(b))); }
  if s == TY_F24 && t == TY_I24 { return new_f24(f32(get_i24(b))); }
  if s == TY_F24 && t == TY_F24 { return b; }
  return new_u24(0u);
}

fn partial_app(a: u32, b: u32) -> u32 {
  return (b & 0xFFFFFFE0u) | get_sym(a);
}

fn u_div(a: u32, b: u32) -> u32 { if b == 0u { return 0u; } return a / b; }
fn u_rem(a: u32, b: u32) -> u32 { if b == 0u { return 0u; } return a % b; }

fn i_div(a: i32, b: i32) -> i32 {
  if b == 0 { return 0; }
  if a == -2147483648 && b == -1 { return 0; }
  return a / b;
}
fn i_rem(a: i32, b: i32) -> i32 {
  if b == 0 { return 0; }
  if a == -2147483648 && b == -1 { return 0; }
  return a % b;
}

fn operate_u24(op: u32, av: u32, bv: u32) -> u32 {
  switch op {
    case 4u: { return new_u24(av + bv); }
    case 5u: { return new_u24(av - bv); }
    case 6u: { return new_u24(bv - av); }
    case 7u: { return new_u24(av * bv); }
    case 8u: { return new_u24(u_div(av, bv)); }
    case 9u: { return new_u24(u_div(bv, av)); }
    case 10u: { return new_u24(u_rem(av, bv)); }
    case 11u: { return new_u24(u_rem(bv, av)); }
    case 12u: { return new_u24(u32(av == bv)); }
    case 13u: { return new_u24(u32(av != bv)); }
    case 14u: { return new_u24(u32(av < bv)); }
    case 15u: { return new_u24(u32(av > bv)); }
    case 16u: { return new_u24(av & bv); }
    case 17u: { return new_u24(av | bv); }
    case 18u: { return new_u24(av ^ bv); }
    case 19u: { return new_u24(av << (bv & 31u)); }
    case 21u: { return new_u24(av >> (bv & 31u)); }
    case 20u: { return new_u24(bv << (av & 31u)); }
    case 22u: { return new_u24(bv >> (av & 31u)); }
    default: { return new_u24(0u); }
  }
}

fn operate_i24(op: u32, av: i32, bv: i32) -> u32 {
  switch op {
    case 4u: { return new_i24(av + bv); }
    case 5u: { return new_i24(av - bv); }
    case 6u: { return new_i24(bv - av); }
    case 7u: { return new_i24(av * bv); }
    case 8u: { return new_i24(i_div(av, bv)); }
    case 9u: { return new_i24(i_div(bv, av)); }
    case 10u: { return new_i24(i_rem(av, bv)); }
    case 11u: { return new_i24(i_rem(bv, av)); }
    case 12u: { return new_u24(u32(av == bv)); }
    case 13u: { return new_u24(u32(av != bv)); }
    case 14u: { return new_u24(u32(av < bv)); }
    case 15u: { return new_u24(u32(av > bv)); }
    case 16u: { return new_i24(av & bv); }
    case 17u: { return new_i24(av | bv); }
    case 18u: { return new_i24(av ^ bv); }
    default: { return new_u24(0u); }
  }
}

fn operate_f24(op: u32, av: f32, bv: f32) -> u32 {
  switch op {
    case 4u: { return new_f24(av + bv); }
    case 5u: { return new_f24(av - bv); }
    case 6u: { return new_f24(bv - av); }
    case 7u: { return new_f24(av * bv); }
    case 8u: { return new_f24(av / bv); }
    case 9u: { return new_f24(bv / av); }
    case 10u: { return new_f24(av % bv); }
    case 11u: { return new_f24(bv % av); }
    case 12u: { return new_u24(u32(av == bv)); }
    case 13u: { return new_u24(u32(av != bv)); }
    case 14u: { return new_u24(u32(av < bv)); }
    case 15u: { return new_u24(u32(av > bv)); }
    case 16u: { return new_f24(atan2(av, bv)); }
    case 17u: { return new_f24(log(bv) / log(av)); }
    case 18u: { return new_f24(pow(av, bv)); }
    case 19u: { return new_f24(sin(av + bv)); }
    case 21u: { return new_f24(tan(av + bv)); }
    default: { return new_u24(0u); }
  }
}

fn operate(a: u32, b: u32) -> u32 {
  let at = get_typ(a);
  let bt = get_typ(b);
  if at == TY_SYM && bt == TY_SYM { return new_u24(0u); }
  if is_cast(a) && is_num(b) { return numb_cast(a, b); }
  if is_cast(b) && is_num(a) { return numb_cast(b, a); }
  if at == TY_SYM && bt != TY_SYM { return partial_app(a, b); }
  if at != TY_SYM && bt == TY_SYM { return partial_app(b, a); }
  if at >= OP_ADD && bt >= OP_ADD { return new_u24(0u); }
  if at < OP_ADD && bt < OP_ADD { return new_u24(0u); }
  var op = at;
  var aa = a;
  var ty = bt;
  var bb = b;
  if at < OP_ADD {
    op = bt; aa = b; ty = at; bb = a;
  }
  if ty == TY_U24 { return operate_u24(op, get_u24(aa), get_u24(bb)); }
  if ty == TY_I24 { return operate_i24(op, get_i24(aa), get_i24(bb)); }
  if ty == TY_F24 { return operate_f24(op, get_f24(aa), get_f24(bb)); }
  return new_u24(0u);
}

fn interact_oper(a: u32, b: u32) -> bool {
  if !get_resources(1u, 0u) { return false; }
  let bi = val_of(b);
  if bi >= atomicLoad(&ctl[M_NLEN]) || node_empty(bi) { return false; }
  let av = val_of(a);
  let bp = node_take(bi);
  let b1 = bp.x;
  let b2 = enter(bp.y);
  if tag_of(b1) == NUM {
    let cv = operate(av, val_of(b1));
    link(mk_port(NUM, cv), b2);
  } else {
    node_store(nloc[0], vec2<u32>(mk_port(tag_of(a), av), b2));
    link(b1, mk_port(OPR, nloc[0]));
  }
  return true;
}

fn interact_swit(a: u32, b: u32) -> bool {
  if !get_resources(2u, 0u) { return false; }
  let bi = val_of(b);
  if bi >= atomicLoad(&ctl[M_NLEN]) || node_empty(bi) { return false; }
  let av = get_u24(val_of(a));
  let bp = node_take(bi);
  if av == 0u {
    node_store(nloc[0], vec2<u32>(bp.y, mk_port(ERA, 0u)));
    link(mk_port(CON, nloc[0]), bp.x);
  } else {
    node_store(nloc[0], vec2<u32>(mk_port(ERA, 0u), mk_port(CON, nloc[1])));
    node_store(nloc[1], vec2<u32>(mk_port(NUM, new_u24(av - 1u)), bp.y));
    link(mk_port(CON, nloc[0]), bp.x);
  }
  return true;
}

fn interact_call(a: u32, b: u32) -> bool {
  let fid = val_of(a) & 0x0FFFFFFFu;
  let ndefs = book[0];
  if fid >= ndefs {
    atomicStore(&ctl[M_ERR], 1u);
    return false;
  }
  let base = 1u + fid * 8u;
  let safe = book[base];
  let rbag_len = book[base + 1u];
  let node_len = book[base + 2u];
  let nvars = book[base + 3u];
  let root = book[base + 4u];
  let rbag_off = book[base + 5u];
  let node_off = book[base + 6u];

  if tag_of(b) == DUP {
    if safe != 0u {
      return interact_eras(a, b);
    }
    atomicStore(&ctl[M_ERR], 2u);
    return false;
  }

  if node_len > MAX_SLOTS || nvars > MAX_SLOTS {
    atomicStore(&ctl[M_ERR], 3u);
    return false;
  }
  if !get_resources(node_len, nvars) { return false; }

  for (var i = 0u; i < nvars; i++) {
    atomicStore(&vars[vloc[i]], NONE);
  }
  for (var i = 0u; i < node_len; i++) {
    let lo = adjust_port(book[node_off + i * 2u]);
    let hi = adjust_port(book[node_off + i * 2u + 1u]);
    node_store(nloc[i], vec2<u32>(lo, hi));
  }
  for (var i = 0u; i < rbag_len; i++) {
    let lo = adjust_port(book[rbag_off + i * 2u]);
    let hi = adjust_port(book[rbag_off + i * 2u + 1u]);
    link(lo, hi);
  }
  link(adjust_port(root), b);
  return true;
}

fn interact_one(aa: u32, bb: u32) -> bool {
  var a = aa;
  var b = bb;
  var rule = get_rule(a, b);
  if tag_of(a) == REF && b == ROOT {
    rule = CALL;
  } else if tag_of(b) < tag_of(a) {
    let x = a; a = b; b = x;
  }

  var ok = false;
  switch rule {
    case 0u: { ok = interact_link(a, b); }
    case 1u: { ok = interact_call(a, b); }
    case 2u: { ok = interact_void(); }
    case 3u: { ok = interact_eras(a, b); }
    case 4u: { ok = interact_anni(a, b); }
    case 5u: { ok = interact_comm(a, b); }
    case 6u: { ok = interact_oper(a, b); }
    case 7u: { ok = interact_swit(a, b); }
    default: { ok = false; }
  }

  if !ok {
    // Transient: node not ready or partition miss. Push back; do not kill the run.
    emit(a, b);
    return false;
  }
  if rule != LINK {
    litrs = litrs + 1u;
  }
  return true;
}

const WG: u32 = 64u;

@compute @workgroup_size(64)
fn evaluator(@builtin(global_invocation_id) gid: vec3<u32>) {
  let nthreads = atomicLoad(&ctl[M_NTHREADS]);
  g_tid = gid.x;
  if g_tid >= nthreads { return; }
  nput = atomicLoad(&worker[wf(g_tid, W_NPUT)]);
  vput = atomicLoad(&worker[wf(g_tid, W_VPUT)]);
  nwrap = atomicLoad(&worker[wf(g_tid, W_NWRAP)]);
  vwrap = atomicLoad(&worker[wf(g_tid, W_VWRAP)]);
  ltop = 0u;
  litrs = 0u;
  let max_steps = atomicLoad(&ctl[M_MAX]);
  for (var s = 0u; s < max_steps; s++) {
    if atomicLoad(&ctl[M_OOM]) != 0u { break; }
    if atomicLoad(&ctl[M_ERR]) != 0u { break; }
    let rd = pop_redex();
    if rd.z == 0u { break; }
    // Failed interact re-emits; keep going (do not treat as idle).
    interact_one(rd.x, rd.y);
  }
  flush_local();
  if litrs > 0u {
    atomicAdd(&ctl[M_ITRS], litrs);
  }
  atomicStore(&worker[wf(g_tid, W_NPUT)], nput);
  atomicStore(&worker[wf(g_tid, W_VPUT)], vput);
  atomicStore(&worker[wf(g_tid, W_NWRAP)], nwrap);
  atomicStore(&worker[wf(g_tid, W_VWRAP)], vwrap);
  atomicAdd(&ctl[M_RLEN], atomicLoad(&worker[wf(g_tid, W_RLEN)]) + ltop);
}
