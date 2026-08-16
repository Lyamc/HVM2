use crate::hvm;

use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Target { CUDA, C, Rust }

// Compiles a whole Book.
pub fn compile_book(trg: Target, book: &hvm::Book) -> String {
  let mut code = String::new();

  // Compiles functions
  for fid in 0..book.defs.len() {
    compile_def(trg, &mut code, book, 0, fid as hvm::Val);
    code.push_str(&format!("\n"));
  }

  // Compiles interact_call dispatcher
  if trg == Target::CUDA {
    code.push_str("__device__ ");
  }
  if trg == Target::Rust {
    code.push_str("fn interact_call_compiled(net: &GNet, tm: &mut TMem, a: Port, b: Port) -> bool {\n");
    code.push_str("  match get_val(a) & 0xFFFFFFF {\n");
    for (fid, def) in book.defs.iter().enumerate() {
      let fun = sanitize_name(&def.name);
      code.push_str(&format!("    {fid} => interact_call_{fun}(net, tm, a, b),\n"));
    }
    code.push_str("    _ => false,\n");
    code.push_str("  }\n");
    code.push_str("}\n");
  } else {
    code.push_str("bool interact_call(Net *net, TM *tm, Port a, Port b) {\n");
    code.push_str("  u32 fid = get_val(a) & 0xFFFFFFF;\n");
    code.push_str("  switch (fid) {\n");
    for (fid, def) in book.defs.iter().enumerate() {
      let fun = sanitize_name(&def.name);
      code.push_str(&format!("    case {fid}: return interact_call_{fun}(net, tm, a, b);\n"));
    }
    code.push_str("    default: return false;\n");
    code.push_str("  }\n");
    code.push_str("}");
  }

  return code;
}

// Compiles a single Def.
pub fn compile_def(trg: Target, code: &mut String, book: &hvm::Book, tab: usize, fid: hvm::Val) {
  let def = &book.defs[fid as usize];
  let fun = sanitize_name(&def.name);

  // Initializes context
  let neo = &mut 0;
  
  // Generates function
  if trg == Target::CUDA {
    code.push_str("__device__ ");
  }
  if trg == Target::Rust {
    code.push_str(&format!("{}fn interact_call_{}(net: &GNet, tm: &mut TMem, a: Port, mut b: Port) -> bool {{\n", indent(tab), fun));
  } else {
    code.push_str(&format!("{}bool interact_call_{}(Net *net, TM *tm, Port a, Port b) {{\n", indent(tab), fun));
  }
  // Fast DUP-REF
  if def.safe {
    code.push_str(&format!("{}if (get_tag(b) == DUP) {{\n", indent(tab+1)));
    code.push_str(&format!("{}return interact_eras(net, tm, a, b);\n", indent(tab+2)));
    code.push_str(&format!("{}}}\n", indent(tab+1)));
  }
  if trg == Target::Rust {
    code.push_str(&format!("{}let mut vl: u32 = 0;\n", indent(tab+1)));
    code.push_str(&format!("{}let mut nl: u32 = 0;\n", indent(tab+1)));
  } else {
    code.push_str(&format!("{}u32 vl = 0;\n", indent(tab+1)));
    code.push_str(&format!("{}u32 nl = 0;\n", indent(tab+1)));
  }

  // Allocs resources (using fast allocator)
  for i in 0 .. def.vars {
    if trg == Target::Rust {
      code.push_str(&format!("{}let v{:x} = vars_alloc_1(net, tm, &mut vl);\n", indent(tab+1), i));
    } else {
      code.push_str(&format!("{}Val v{:x} = vars_alloc_1(net, tm, &vl);\n", indent(tab+1), i));
    }
  }
  for i in 0 .. def.node.len() {
    if trg == Target::Rust {
      code.push_str(&format!("{}let n{:x} = node_alloc_1(net, tm, &mut nl);\n", indent(tab+1), i));
    } else {
      code.push_str(&format!("{}Val n{:x} = node_alloc_1(net, tm, &nl);\n", indent(tab+1), i));
    }
  }
  if trg == Target::Rust {
    code.push_str(&format!("{}if false", indent(tab+1)));
    for i in 0 .. def.vars {
      code.push_str(&format!(" || v{:x} == 0", i));
    }
    for i in 0 .. def.node.len() {
      code.push_str(&format!(" || n{:x} == 0", i));
    }
    code.push_str(" {\n");
  } else {
    code.push_str(&format!("{}if (0", indent(tab+1)));
    for i in 0 .. def.vars {
      code.push_str(&format!(" || !v{:x}", i));
    }
    for i in 0 .. def.node.len() {
      code.push_str(&format!(" || !n{:x}", i));
    }
    code.push_str(") {\n");
  }
  code.push_str(&format!("{}return false;\n", indent(tab+2)));
  code.push_str(&format!("{}}}\n", indent(tab+1)));
  for i in 0 .. def.vars {
    code.push_str(&format!("{}vars_create(net, v{:x}, NONE);\n", indent(tab+1), i));
  }

  // Allocs resources (using slow allocator)
  //code.push_str(&format!("{}// Allocates needed resources.\n", indent(tab+1)));
  //code.push_str(&format!("{}if (!get_resources(net, tm, {}, {}, {})) {{\n", indent(tab+1), def.rbag.len()+1, def.node.len(), def.vars));
  //code.push_str(&format!("{}return false;\n", indent(tab+2)));
  //code.push_str(&format!("{}}}\n", indent(tab+1)));
  //for i in 0 .. def.node.len() {
    //code.push_str(&format!("{}Val n{:x} = tm->nloc[0x{:x}];\n", indent(tab+1), i, i));
  //}
  //for i in 0 .. def.vars {
    //code.push_str(&format!("{}Val v{:x} = tm->vloc[0x{:x}];\n", indent(tab+1), i, i));
  //}
  //for i in 0 .. def.vars {
    //code.push_str(&format!("{}vars_create(net, v{:x}, NONE);\n", indent(tab+1), i));
  //}

  // Compiles root
  compile_link_fast(trg, code, book, neo, tab+1, def, def.root, "b");

  // Compiles rbag
  for redex in &def.rbag {
    let fun = compile_node(trg, code, book, neo, tab+1, def, redex.get_fst());
    let arg = compile_node(trg, code, book, neo, tab+1, def, redex.get_snd());
    code.push_str(&format!("{}link(net, tm, {}, {});\n", indent(tab+1), &fun, &arg));
  }

  // Return
  code.push_str(&format!("{}return true;\n", indent(tab+1)));
  code.push_str(&format!("{}}}\n", indent(tab)));
}

// Compiles a link, performing some pre-defined static reductions.
pub fn compile_link_fast(trg: Target, code: &mut String, book: &hvm::Book, neo: &mut usize, tab: usize, def: &hvm::Def, a: hvm::Port, b: &str) {

  // (<?(a111 a112) a12> a2) <~ (#X R)
  // --------------------------------- fast SWITCH
  // if X == 0:
  //   a111 <~ R
  //   a112 <~ ERAS
  // else:
  //   a111 <~ ERAS
  //   a112 <~ (#(X-1) R)
  if trg != Target::CUDA && a.get_tag() == hvm::CON {
    let a_ = &def.node[a.get_val() as usize];
    let a1 = a_.get_fst();
    let a2 = a_.get_snd();
    if a1.get_tag() == hvm::SWI {
      let a1_ = &def.node[a1.get_val() as usize];
      let a11 = a1_.get_fst();
      let a12 = a1_.get_snd();
      if a11.get_tag() == hvm::CON && a2.get_tag() == hvm::VAR && a12.0 == a2.0 {
        let a11_ = &def.node[a11.get_val() as usize];
        let a111 = a11_.get_fst();
        let a112 = a11_.get_snd();
        let op   = fresh(neo);
        let bv   = fresh(neo);
        let x1   = fresh(neo);
        let x2   = fresh(neo);
        let nu   = fresh(neo);
        code.push_str(&format!("{}{}\n", indent(tab), decl_bool(trg, &op)));
        code.push_str(&format!("{}{}\n", indent(tab), decl_pair(trg, &bv)));
        code.push_str(&format!("{}{}\n", indent(tab), decl_port(trg, &nu)));
        code.push_str(&format!("{}{}\n", indent(tab), decl_port(trg, &x1)));
        code.push_str(&format!("{}{}\n", indent(tab), decl_port(trg, &x2)));
        code.push_str(&format!("{}//fast switch\n", indent(tab)));
        code.push_str(&format!("{}if (get_tag({}) == CON) {{\n", indent(tab), b));
        code.push_str(&format!("{}{} = node_load(net, get_val({}));\n", indent(tab+1), &bv, b)); // recycled
        code.push_str(&format!("{}{} = enter(net,get_fst({}));\n", indent(tab+1), &nu, &bv));
        code.push_str(&format!("{}if (get_tag({}) == NUM) {{\n", indent(tab+1), &nu));
        code.push_str(&format!("{}{} += 3;\n", indent(tab+2), tm_itrs(trg)));
        code.push_str(&format!("{}vars_take(net, v{:x});\n", indent(tab+2), a2.get_val()));
        code.push_str(&format!("{}{} = {};\n", indent(tab+2), &op, lit_true(trg)));
        code.push_str(&format!("{}if (get_u24(get_val({})) == 0) {{\n", indent(tab+2), &nu));
        code.push_str(&format!("{}node_take(net, get_val({}));\n", indent(tab+3), b));
        code.push_str(&format!("{}{} = get_snd({});\n", indent(tab+3), &x1, &bv));
        code.push_str(&format!("{}{} = new_port(ERA,0);\n", indent(tab+3), &x2));
        code.push_str(&format!("{}}} else {{\n", indent(tab+2)));
        code.push_str(&format!("{}node_store(net, get_val({}), new_pair(new_port(NUM,new_u24(get_u24(get_val({}))-1)), get_snd({})));\n", indent(tab+3), b, &nu, &bv));
        code.push_str(&format!("{}{} = new_port(ERA,0);\n", indent(tab+3), &x1));
        code.push_str(&format!("{}{} = {};\n", indent(tab+3), &x2, b));
        code.push_str(&format!("{}}}\n", indent(tab+2)));
        code.push_str(&format!("{}}} else {{\n", indent(tab+1)));
        code.push_str(&format!("{}node_store(net, get_val({}), new_pair({},get_snd({})));\n", indent(tab+2), b, &nu, &bv)); // update "entered" var
        code.push_str(&format!("{}}}\n", indent(tab+1)));
        code.push_str(&format!("{}}}\n", indent(tab+0)));
        compile_link_fast(trg, code, book, neo, tab, def, a111, &x1);
        compile_link_fast(trg, code, book, neo, tab, def, a112, &x2);
        code.push_str(&format!("{}if (!{}) {{\n", indent(tab), &op));
        code.push_str(&format!("{}node_create(net, n{:x}, new_pair(new_port(SWI,n{:x}),new_port(VAR,v{:x})));\n", indent(tab+1), a.get_val(), a1.get_val(), a2.get_val()));
        code.push_str(&format!("{}node_create(net, n{:x}, new_pair(new_port(CON,n{:x}),new_port(VAR,v{:x})));\n", indent(tab+1), a1.get_val(), a11.get_val(), a12.get_val()));
        code.push_str(&format!("{}node_create(net, n{:x}, new_pair({},{}));\n", indent(tab+1), a11.get_val(), &x1, &x2));
        link_or_store(trg, code, book, neo, tab+1, def, &format!("new_port(CON, n{:x})", a.get_val()), b);
        code.push_str(&format!("{}}}\n", indent(tab)));
        return;
      }
    }
  }

  // FIXME: REVIEW
  // <+ #B r> <~ #A
  // --------------- fast OPER
  // r <~ #(op(A,B))
  if trg != Target::CUDA && a.get_tag() == hvm::OPR {
    let a_ = &def.node[a.get_val() as usize];
    let a1 = a_.get_fst();
    let a2 = a_.get_snd();
    let op = fresh(neo);
    let x1 = compile_node(trg, code, book, neo, tab, def, a1);
    let x2 = fresh(neo);
    code.push_str(&format!("{}{}\n", indent(tab), decl_bool(trg, &op)));
    code.push_str(&format!("{}{}\n", indent(tab), decl_port(trg, &x2)));
    code.push_str(&format!("{}// fast oper\n", indent(tab)));
    code.push_str(&format!("{}if (get_tag({}) == NUM && get_tag({}) == NUM) {{\n", indent(tab), b, &x1));
    code.push_str(&format!("{}{} += 1;\n", indent(tab+1), tm_itrs(trg)));
    code.push_str(&format!("{}{} = {};\n", indent(tab+1), &op, lit_true(trg)));
    code.push_str(&format!("{}{} = new_port(NUM, operate(get_val({}), get_val({})));\n", indent(tab+1), &x2, b, &x1));
    code.push_str(&format!("{}}}\n", indent(tab)));
    compile_link_fast(trg, code, book, neo, tab, def, a2, &x2);
    code.push_str(&format!("{}if (!{}) {{\n", indent(tab), &op));
    code.push_str(&format!("{}node_create(net, n{:x}, new_pair({},{}));\n", indent(tab+1), a.get_val(), &x1, &x2));
    link_or_store(trg, code, book, neo, tab+1, def, &format!("new_port(OPR, n{:x})", a.get_val()), b);
    code.push_str(&format!("{}}}\n", indent(tab)));
    return;
  }

  // FIXME: REVIEW
  // {a1 a2} <~ #v
  // ------------- Fast COPY
  // a1 <~ #v
  // a2 <~ #v
  if trg != Target::CUDA && a.get_tag() == hvm::DUP {
    let a_ = &def.node[a.get_val() as usize];
    let p1 = a_.get_fst();
    let p2 = a_.get_snd();
    let op = fresh(neo);
    let x1 = fresh(neo);
    let x2 = fresh(neo);
    code.push_str(&format!("{}{}\n", indent(tab), decl_bool(trg, &op)));
    code.push_str(&format!("{}{}\n", indent(tab), decl_port(trg, &x1)));
    code.push_str(&format!("{}{}\n", indent(tab), decl_port(trg, &x2)));
    code.push_str(&format!("{}// fast copy\n", indent(tab)));
    code.push_str(&format!("{}if (get_tag({}) == NUM) {{\n", indent(tab), b));
    code.push_str(&format!("{}{} += 1;\n", indent(tab+1), tm_itrs(trg)));
    code.push_str(&format!("{}{} = {};\n", indent(tab+1), &op, lit_true(trg)));
    code.push_str(&format!("{}{} = {};\n", indent(tab+1), &x1, b));
    code.push_str(&format!("{}{} = {};\n", indent(tab+1), &x2, b));
    code.push_str(&format!("{}}}\n", indent(tab)));
    compile_link_fast(trg, code, book, neo, tab, def, p2, &x2);
    compile_link_fast(trg, code, book, neo, tab, def, p1, &x1);
    code.push_str(&format!("{}if (!{}) {{\n", indent(tab), &op));
    code.push_str(&format!("{}node_create(net, n{:x}, new_pair({},{}));\n", indent(tab+1), a.get_val(), x1, x2));
    link_or_store(trg, code, book, neo, tab+1, def, &format!("new_port(DUP,n{:x})", a.get_val()), b);
    code.push_str(&format!("{}}}\n", indent(tab)));
    return;
  }

  // (a1 a2) <~ (x1 x2)
  // ------------------ Fast ANNI
  // a1 <~ x1
  // a2 <~ x2
  if trg != Target::CUDA && a.get_tag() == hvm::CON {
    let a_ = &def.node[a.get_val() as usize];
    let a1 = a_.get_fst();
    let a2 = a_.get_snd();
    let op = fresh(neo);
    let bv = fresh(neo);
    let x1 = fresh(neo);
    let x2 = fresh(neo);
    code.push_str(&format!("{}{}\n", indent(tab), decl_bool(trg, &op)));
    code.push_str(&format!("{}{}\n", indent(tab), decl_pair(trg, &bv)));
    code.push_str(&format!("{}{}\n", indent(tab), decl_port(trg, &x1)));
    code.push_str(&format!("{}{}\n", indent(tab), decl_port(trg, &x2)));
    code.push_str(&format!("{}// fast anni\n", indent(tab)));
    code.push_str(&format!("{}if (get_tag({}) == CON && {}) {{\n", indent(tab), b, node_nz(trg, b)));
    //code.push_str(&format!("{}atomic_fetch_add(&FAST, 1);\n", indent(tab+1)));
    code.push_str(&format!("{}{} += 1;\n", indent(tab+1), tm_itrs(trg)));
    code.push_str(&format!("{}{} = {};\n", indent(tab+1), &op, lit_true(trg)));
    code.push_str(&format!("{}{} = node_take(net, get_val({}));\n", indent(tab+1), &bv, b));
    code.push_str(&format!("{}{} = get_fst({});\n", indent(tab+1), x1, &bv));
    code.push_str(&format!("{}{} = get_snd({});\n", indent(tab+1), x2, &bv));
    code.push_str(&format!("{}}}\n", indent(tab)));
    //code.push_str(&format!("{}else {{ atomic_fetch_add(&SLOW, 1); }}\n", indent(tab)));
    compile_link_fast(trg, code, book, neo, tab, def, a2, &x2);
    compile_link_fast(trg, code, book, neo, tab, def, a1, &x1);
    code.push_str(&format!("{}if (!{}) {{\n", indent(tab), &op));
    code.push_str(&format!("{}node_create(net, n{:x}, new_pair({},{}));\n", indent(tab+1), a.get_val(), x1, x2));
    link_or_store(trg, code, book, neo, tab+1, def, &format!("new_port(CON,n{:x})", a.get_val()), b);
    code.push_str(&format!("{}}}\n", indent(tab)));
    return;
  }

  // FIXME: since get_tag(NONE) == REF, comparing if something's tag is REF always has the issue of
  // returning true when that thing is NONE. this caused a bug in the optimization below. in
  // general, this is a potential source of bugs across the entire implementation, so we always
  // need to check that case. an alternative, of course, would be to make get_tag handle this, but
  // I'm concerned about the performance issues. so, instead, we should make sure that, across the
  // entire codebase, we never use get_tag expecting a REF on something that might be NONE
  
  // ATOM <~ *
  // --------- Fast VOID
  // nothing
  if trg != Target::CUDA && (a.get_tag() == hvm::NUM || a.get_tag() == hvm::ERA) {
    code.push_str(&format!("{}// fast void\n", indent(tab)));
    code.push_str(&format!("{}if (get_tag({}) == ERA || get_tag({}) == NUM) {{\n", indent(tab), b, b));
    code.push_str(&format!("{}{} += 1;\n", indent(tab+1), tm_itrs(trg)));
    code.push_str(&format!("{}}} else {{\n", indent(tab)));
    compile_link_slow(trg, code, book, neo, tab+1, def, a, b);
    code.push_str(&format!("{}}}\n", indent(tab)));
    return;
  }

  compile_link_slow(trg, code, book, neo, tab, def, a, b);
}

// Compiles a link, without pre-defined reductions.
pub fn compile_link_slow(trg: Target, code: &mut String, book: &hvm::Book, neo: &mut usize, tab: usize, def: &hvm::Def, a: hvm::Port, b: &str) {
  let a_node = compile_node(trg, code, book, neo, tab, def, a);
  link_or_store(trg, code, book, neo, tab, def, &a_node, b);
}

// TODO: comment
pub fn link_or_store(trg: Target, code: &mut String, book: &hvm::Book, neo: &mut usize, tab: usize, def: &hvm::Def, a: &str, b: &str) {
  code.push_str(&format!("{}if ({} != NONE) {{\n", indent(tab), b));
  code.push_str(&format!("{}link(net, tm, {}, {});\n", indent(tab+1), a, b));
  code.push_str(&format!("{}}} else {{\n", indent(tab)));
  code.push_str(&format!("{}{} = {};\n", indent(tab+1), b, a));
  code.push_str(&format!("{}}}\n", indent(tab)));
}

// Compiles just a node.
pub fn compile_node(trg: Target, code: &mut String, book: &hvm::Book, neo: &mut usize, tab: usize, def: &hvm::Def, a: hvm::Port) -> String {
  if a.is_nod() {
    let nd = &def.node[a.get_val() as usize];
    let p1 = compile_node(trg, code, book, neo, tab, def, nd.get_fst());
    let p2 = compile_node(trg, code, book, neo, tab, def, nd.get_snd());
    code.push_str(&format!("{}node_create(net, n{:x}, new_pair({},{}));\n", indent(tab), a.get_val(), p1, p2));
    return format!("new_port({},n{:x})", compile_tag(trg, a.get_tag()), a.get_val());
  } else if a.is_var() {
    return format!("new_port(VAR,v{:x})", a.get_val());
  } else {
    return format!("new_port({},0x{:08x})", compile_tag(trg, a.get_tag()), a.get_val());
  }
}

// Compiles an atomic port.
//fn compile_atom(trg: Target, port: hvm::Port) -> String {
  //return format!("new_port({},0x{:08x})/*atom*/", compile_tag(trg, port.get_tag()), port.get_val());
//}

fn sanitize_name(name: &str) -> String {
  name.replace("/", "_").replace(".", "_").replace("-", "_")
}

fn tm_itrs(trg: Target) -> &'static str {
  if trg == Target::Rust { "tm.itrs" } else { "tm->itrs" }
}

fn lit_true(trg: Target) -> &'static str {
  if trg == Target::Rust { "true" } else { "1" }
}

fn decl_bool(trg: Target, name: &str) -> String {
  if trg == Target::Rust {
    format!("let mut {name}: bool = false;")
  } else {
    format!("bool {name} = 0;")
  }
}

fn decl_pair(trg: Target, name: &str) -> String {
  if trg == Target::Rust {
    format!("let mut {name}: Pair = Pair(0);")
  } else {
    format!("Pair {name} = 0;")
  }
}

fn decl_port(trg: Target, name: &str) -> String {
  if trg == Target::Rust {
    format!("let mut {name}: Port = NONE;")
  } else {
    format!("Port {name} = NONE;")
  }
}

fn node_nz(trg: Target, b: &str) -> String {
  if trg == Target::Rust {
    format!("node_load(net, get_val({b})).0 != 0")
  } else {
    format!("node_load(net, get_val({b})) != 0")
  }
}

// Compiles a tag.
pub fn compile_tag(trg: Target, tag: hvm::Tag) -> &'static str {
  match tag {
    hvm::VAR => "VAR",
    hvm::REF => "REF",
    hvm::ERA => "ERA",
    hvm::NUM => "NUM",
    hvm::OPR => "OPR",
    hvm::SWI => "SWI",
    hvm::CON => "CON",
    hvm::DUP => "DUP",
    _ => unreachable!(),
  }
}

// Creates indentation.
pub fn indent(tab: usize) -> String {
  return "  ".repeat(tab);
}

// Generates a fresh name.
fn fresh(count: &mut usize) -> String {
  *count += 1;
  format!("k{}", count)
}
