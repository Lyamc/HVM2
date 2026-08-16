use crate::hvm;
use std::collections::BTreeMap;

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Numb(pub u32);

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum Tree {
  Var { nam: String },
  Ref { nam: String },
  Era,
  Num { val: Numb },
  Con { fst: Box<Tree>, snd: Box<Tree> },
  Dup { fst: Box<Tree>, snd: Box<Tree> },
  Opr { fst: Box<Tree>, snd: Box<Tree> },
  Swi { fst: Box<Tree>, snd: Box<Tree> },
}

pub type Redex = (bool, Tree, Tree);

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Net {
  pub root: Tree,
  pub rbag: Vec<Redex>,
}

impl Numb {
  pub fn show(&self) -> String {
    let numb = hvm::Numb(self.0);
    match numb.get_typ() {
      hvm::TY_SYM => match numb.get_sym() as hvm::Tag {
        hvm::TY_U24 => "[u24]".to_string(),
        hvm::TY_I24 => "[i24]".to_string(),
        hvm::TY_F24 => "[f24]".to_string(),
        hvm::OP_ADD => "[+]".to_string(),
        hvm::OP_SUB => "[-]".to_string(),
        hvm::FP_SUB => "[:-]".to_string(),
        hvm::OP_MUL => "[*]".to_string(),
        hvm::OP_DIV => "[/]".to_string(),
        hvm::FP_DIV => "[:/]".to_string(),
        hvm::OP_REM => "[%]".to_string(),
        hvm::FP_REM => "[:%]".to_string(),
        hvm::OP_EQ => "[=]".to_string(),
        hvm::OP_NEQ => "[!]".to_string(),
        hvm::OP_LT => "[<]".to_string(),
        hvm::OP_GT => "[>]".to_string(),
        hvm::OP_AND => "[&]".to_string(),
        hvm::OP_OR => "[|]".to_string(),
        hvm::OP_XOR => "[^]".to_string(),
        hvm::OP_SHL => "[<<]".to_string(),
        hvm::FP_SHL => "[:<<]".to_string(),
        hvm::OP_SHR => "[>>]".to_string(),
        hvm::FP_SHR => "[:>>]".to_string(),
        _ => "[?]".to_string(),
      },
      hvm::TY_U24 => format!("{}", numb.get_u24()),
      hvm::TY_I24 => format!("{:+}", numb.get_i24()),
      hvm::TY_F24 => {
        let val = numb.get_f24();
        if val.is_infinite() {
          if val.is_sign_positive() { "+inf".into() } else { "-inf".into() }
        } else if val.is_nan() {
          "+NaN".into()
        } else {
          format!("{:?}", val)
        }
      }
      _ => {
        let typ = numb.get_typ();
        let val = numb.get_u24();
        format!("[{}0x{:07X}]", match typ {
          hvm::OP_ADD => "+",
          hvm::OP_SUB => "-",
          hvm::FP_SUB => ":-",
          hvm::OP_MUL => "*",
          hvm::OP_DIV => "/",
          hvm::FP_DIV => ":/",
          hvm::OP_REM => "%",
          hvm::FP_REM => ":%",
          hvm::OP_EQ => "=",
          hvm::OP_NEQ => "!",
          hvm::OP_LT => "<",
          hvm::OP_GT => ">",
          hvm::OP_AND => "&",
          hvm::OP_OR => "|",
          hvm::OP_XOR => "^",
          hvm::OP_SHL => "<<",
          hvm::FP_SHL => ":<<",
          hvm::OP_SHR => ">>",
          hvm::FP_SHR => ":>>",
          _ => "?",
        }, val)
      }
    }
  }
}

impl Tree {
  pub fn show(&self) -> String {
    match self {
      Tree::Var { nam } => nam.to_string(),
      Tree::Ref { nam } => format!("@{}", nam),
      Tree::Era => "*".to_string(),
      Tree::Num { val } => val.show(),
      Tree::Con { fst, snd } => format!("({} {})", fst.show(), snd.show()),
      Tree::Dup { fst, snd } => format!("{{{} {}}}", fst.show(), snd.show()),
      Tree::Opr { fst, snd } => format!("$({} {})", fst.show(), snd.show()),
      Tree::Swi { fst, snd } => format!("?({} {})", fst.show(), snd.show()),
    }
  }

  pub fn readback(net: &hvm::GNet, port: hvm::Port, fids: &BTreeMap<hvm::Val, String>) -> Option<Tree> {
    match port.get_tag() {
      hvm::VAR => {
        let got = net.enter(port);
        if got != port {
          Tree::readback(net, got, fids)
        } else {
          Some(Tree::Var { nam: format!("v{:x}", port.get_val()) })
        }
      }
      hvm::REF => Some(Tree::Ref { nam: fids.get(&port.get_val())?.clone() }),
      hvm::ERA => Some(Tree::Era),
      hvm::NUM => Some(Tree::Num { val: Numb(port.get_val()) }),
      hvm::CON => {
        let pair = net.node_load(port.get_val() as usize);
        Some(Tree::Con {
          fst: Box::new(Tree::readback(net, pair.get_fst(), fids)?),
          snd: Box::new(Tree::readback(net, pair.get_snd(), fids)?),
        })
      }
      hvm::DUP => {
        let pair = net.node_load(port.get_val() as usize);
        Some(Tree::Dup {
          fst: Box::new(Tree::readback(net, pair.get_fst(), fids)?),
          snd: Box::new(Tree::readback(net, pair.get_snd(), fids)?),
        })
      }
      hvm::OPR => {
        let pair = net.node_load(port.get_val() as usize);
        Some(Tree::Opr {
          fst: Box::new(Tree::readback(net, pair.get_fst(), fids)?),
          snd: Box::new(Tree::readback(net, pair.get_snd(), fids)?),
        })
      }
      hvm::SWI => {
        let pair = net.node_load(port.get_val() as usize);
        Some(Tree::Swi {
          fst: Box::new(Tree::readback(net, pair.get_fst(), fids)?),
          snd: Box::new(Tree::readback(net, pair.get_snd(), fids)?),
        })
      }
      _ => unreachable!(),
    }
  }
}

impl Net {
  pub fn show(&self) -> String {
    self.root.show()
  }

  pub fn readback(net: &hvm::GNet, book: &hvm::Book) -> Option<Net> {
    let mut fids = BTreeMap::new();
    for (fid, def) in book.defs.iter().enumerate() {
      fids.insert(fid as hvm::Val, def.name.clone());
    }
    let root = Tree::readback(net, net.enter(hvm::ROOT), &fids)?;
    Some(Net { root, rbag: Vec::new() })
  }
}
