//! Constrained decoding: `logit_bias` and llama.cpp's GBNF grammars (the request field `grammar`). A `Constraint`
//! changes the logits before a token is picked, greedy or sampled, and follows the tokens the reply gets.

use anyhow::{Context, bail, ensure};
use serde_json::Value;
use std::sync::Arc;

/// A `logit_bias` entry: a token id, or a text whose every token gets the bias (llama.cpp).
#[derive(Clone, Debug, PartialEq)]
pub enum BiasKey { Id(u32), Text(String) }

/// `logit_bias` as OpenAI (`{"<id>": bias}`) or llama.cpp (`[[id or text, bias or false], ...]`) sends it; a key
/// that is not a number is a text. A bias is in [-100, 100]; -100 and `false` ban the token.
pub fn parse_bias(v: &Value) -> anyhow::Result<Vec<(BiasKey, f32)>> {
    let key = |k: &Value| match k {
        Value::String(s) => Ok(s.parse().map_or_else(|_| BiasKey::Text(s.clone()), BiasKey::Id)),
        x => x.as_u64().and_then(|n| u32::try_from(n).ok()).map(BiasKey::Id).context("logit_bias: a key must be a token id or a text"),
    };
    let bias = |b: &Value| match b {
        Value::Bool(false) => Ok(f32::NEG_INFINITY),
        x => x.as_f64().filter(|f| (-100.0..=100.0).contains(f))
            .map(|f| if f <= -100.0 { f32::NEG_INFINITY } else { f as f32 }).context("logit_bias: a bias must be a number in [-100, 100] or false"),
    };
    match v {
        Value::Null => Ok(vec![]),
        Value::Object(o) => o.iter().map(|(k, b)| Ok((key(&Value::String(k.clone()))?, bias(b)?))).collect(),
        Value::Array(a) => a.iter().map(|e| match e.as_array().map(Vec::as_slice) {
            Some([k, b]) => Ok((key(k)?, bias(b)?)),
            _ => bail!("logit_bias: an entry must be [token, bias]"),
        }).collect(),
        _ => bail!("logit_bias must be an object or an array"),
    }
}

/// One element of a rule's sequence: a character (in or, negated, not in the ranges) or another rule.
#[derive(Clone, Debug, PartialEq)]
enum Elem { Chars(bool, Vec<(u32, u32)>), Rule(usize) }

/// A parsed GBNF grammar: each rule is a list of alternatives, each a sequence; repetitions and groups are rules of
/// their own, as in llama.cpp.
#[derive(Debug)]
pub struct Grammar { rules: Vec<Vec<Vec<Elem>>>, root: usize }

/// Limits for a grammar that comes with a request: its length, the nesting of groups, the elements and rules the
/// parser writes out (repetitions copy their body), and the expansion steps a check may take.
const MAX_CHARS: usize = 64 << 10;
const MAX_NEST: usize = 64;
const MAX_ELEMS: usize = 64 << 10;
/// Expansion steps for one character (`advance`), and for checking every rule start at parse time.
const STEPS: usize = 10_000;
const CHECK_STEPS: usize = 200_000;
/// Stacks kept after a character. ponytail: more are dropped (an odd ambiguous grammar may then refuse a valid text).
const MAX_STACKS: usize = 1024;

struct Parser { c: Vec<char>, i: usize, names: Vec<String>, rules: Vec<Option<Vec<Vec<Elem>>>>, nest: usize, elems: usize }

impl Parser {
    fn peek(&self) -> Option<char> { self.c.get(self.i).copied() }
    fn at(&self, s: &str) -> bool { s.chars().enumerate().all(|(k, ch)| self.c.get(self.i + k) == Some(&ch)) }
    /// Spaces and `#` comments, and line ends when `lines`.
    fn ws(&mut self, lines: bool) {
        while let Some(ch) = self.peek() {
            match ch {
                ' ' | '\t' => self.i += 1,
                '\r' | '\n' if lines => self.i += 1,
                '#' => while self.peek().is_some_and(|c| c != '\n') { self.i += 1 },
                _ => break,
            }
        }
    }
    fn name(&mut self) -> anyhow::Result<String> {
        let s = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') { self.i += 1; }
        ensure!(self.i > s, "grammar: a rule name was expected at character {}", s);
        Ok(self.c[s..self.i].iter().collect())
    }
    fn id(&mut self, name: &str) -> usize {
        self.names.iter().position(|n| n == name).unwrap_or_else(|| { self.names.push(name.into()); self.rules.push(None); self.names.len() - 1 })
    }
    /// Counts `n` more elements written out, and fails past `MAX_ELEMS`.
    fn grow(&mut self, n: usize) -> anyhow::Result<()> {
        self.elems += n;
        ensure!(self.elems <= MAX_ELEMS, "grammar: more than {MAX_ELEMS} elements once repetitions are written out");
        Ok(())
    }
    fn new_rule(&mut self, alts: Vec<Vec<Elem>>) -> usize {
        self.names.push(String::new());
        self.rules.push(Some(alts));
        self.rules.len() - 1
    }
    fn hex(&mut self, n: usize) -> anyhow::Result<u32> {
        let s: String = self.c.get(self.i..self.i + n).context("grammar: a short escape")?.iter().collect();
        self.i += n;
        u32::from_str_radix(&s, 16).ok().context("grammar: a bad hex escape")
    }
    /// One character of a literal or a class, escapes read.
    fn chr(&mut self) -> anyhow::Result<u32> {
        let ch = self.peek().context("grammar: unexpected end")?;
        self.i += 1;
        if ch != '\\' { return Ok(ch as u32); }
        let e = self.peek().context("grammar: unexpected end")?;
        self.i += 1;
        Ok(match e { 'x' => self.hex(2)?, 'u' => self.hex(4)?, 'U' => self.hex(8)?, 'n' => 10, 'r' => 13, 't' => 9, c => c as u32 })
    }
    fn alts(&mut self, nested: bool) -> anyhow::Result<Vec<Vec<Elem>>> {
        let mut out = vec![self.seq(nested)?];
        while self.peek() == Some('|') {
            self.i += 1;
            self.ws(true);
            out.push(self.seq(nested)?);
        }
        Ok(out)
    }
    fn seq(&mut self, nested: bool) -> anyhow::Result<Vec<Elem>> {
        let (mut out, mut last) = (vec![], 0);
        while let Some(ch) = self.peek() {
            let start = out.len();
            match ch {
                '"' => {
                    self.i += 1;
                    while self.peek() != Some('"') { let c = self.chr()?; out.push(Elem::Chars(false, vec![(c, c)])); }
                    self.i += 1;
                }
                '[' => {
                    self.i += 1;
                    let neg = self.peek() == Some('^');
                    if neg { self.i += 1; }
                    let mut r = vec![];
                    while self.peek() != Some(']') {
                        let lo = self.chr()?;
                        let hi = if self.peek() == Some('-') && self.c.get(self.i + 1) != Some(&']') { self.i += 1; self.chr()? } else { lo };
                        r.push((lo, hi));
                    }
                    self.i += 1;
                    out.push(Elem::Chars(neg, r));
                }
                '.' => { self.i += 1; out.push(Elem::Chars(true, vec![])); }
                '(' => {
                    self.i += 1;
                    self.ws(true);
                    self.nest += 1;
                    ensure!(self.nest <= MAX_NEST, "grammar: groups nested more than {MAX_NEST} deep");
                    let alts = self.alts(true)?;
                    self.nest -= 1;
                    ensure!(self.peek() == Some(')'), "grammar: ')' expected at character {}", self.i);
                    self.i += 1;
                    out.push(Elem::Rule(self.new_rule(alts)));
                }
                '*' | '+' | '?' | '{' => {
                    ensure!(last < out.len(), "grammar: '{ch}' with nothing before it at character {}", self.i);
                    self.i += 1;
                    let (min, max) = match ch {
                        '*' => (0, None),
                        '+' => (1, None),
                        '?' => (0, Some(1)),
                        _ => {
                            let num = |p: &mut Parser| { p.ws(false); let s = p.i; while p.peek().is_some_and(|c| c.is_ascii_digit()) { p.i += 1; }
                                p.c[s..p.i].iter().collect::<String>().parse::<usize>().ok() };
                            let min = num(self).context("grammar: a number expected after '{'")?;
                            self.ws(false);
                            let max = if self.peek() == Some(',') { self.i += 1; num(self) } else { Some(min) };
                            self.ws(false);
                            ensure!(self.peek() == Some('}') && max.is_none_or(|m| m >= min), "grammar: a bad {{m,n}} at character {}", self.i);
                            self.i += 1;
                            (min, max)
                        }
                    };
                    let body = out.split_off(last);
                    self.repeat(&mut out, body, min, max)?;
                    self.ws(nested);
                    continue;
                }
                c if c.is_ascii_alphanumeric() || c == '-' || c == '_' => {
                    // `name ::=` starts the next rule.
                    let save = self.i;
                    let n = self.name()?;
                    self.ws(false);
                    if self.at("::=") { self.i = save; break; }
                    let id = self.id(&n);
                    out.push(Elem::Rule(id));
                }
                _ => break,
            }
            last = start;
            self.ws(nested);
        }
        Ok(out)
    }
    /// `body` from `min` to `max` (None: no limit) times, as llama.cpp writes it out.
    fn repeat(&mut self, out: &mut Vec<Elem>, body: Vec<Elem>, min: usize, max: Option<usize>) -> anyhow::Result<()> {
        // The copies, checked before they are made.
        self.grow((body.len() + 1).saturating_mul(max.unwrap_or(min).max(1)))?;
        for _ in 0..min { out.extend(body.iter().cloned()); }
        match max {
            None => {
                // r ::= body r |
                let r = self.new_rule(vec![]);
                let mut a = body;
                a.push(Elem::Rule(r));
                self.rules[r] = Some(vec![a, vec![]]);
                out.push(Elem::Rule(r));
            }
            Some(max) => {
                let mut inner: Option<usize> = None;
                for _ in min..max {
                    let mut a = body.clone();
                    a.extend(inner.map(Elem::Rule));
                    inner = Some(self.new_rule(vec![a, vec![]]));
                }
                out.extend(inner.map(Elem::Rule));
            }
        }
        Ok(())
    }
}

impl Grammar {
    pub fn parse(src: &str) -> anyhow::Result<Grammar> {
        ensure!(src.chars().count() <= MAX_CHARS, "grammar: longer than {MAX_CHARS} characters");
        let mut p = Parser { c: src.chars().collect(), i: 0, names: vec![], rules: vec![], nest: 0, elems: 0 };
        p.ws(true);
        while p.i < p.c.len() {
            let name = p.name()?;
            p.ws(false);
            ensure!(p.at("::="), "grammar: '::=' expected after {name}");
            p.i += 3;
            p.ws(true);
            let id = p.id(&name);
            let alts = p.alts(false)?;
            ensure!(p.rules[id].is_none(), "grammar: rule {name} is defined twice");
            p.rules[id] = Some(alts);
            p.ws(true);
            ensure!(p.i == p.c.len() || p.peek().is_some_and(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
                "grammar: unexpected {:?} at character {}", p.peek().unwrap_or(' '), p.i);
        }
        let root = p.names.iter().position(|n| n == "root").context("grammar: no root rule")?;
        let rules = p.rules.into_iter().zip(&p.names).map(|(r, n)| r.with_context(|| format!("grammar: rule {n} is used but not defined")))
            .collect::<anyhow::Result<Vec<_>>>()?;
        ensure!(rules.iter().flatten().map(Vec::len).sum::<usize>() <= MAX_ELEMS, "grammar: more than {MAX_ELEMS} elements");
        let g = Grammar { rules, root };
        // Every rule start expands within one budget, so a grammar whose expansion blows up is refused here and not
        // while it runs.
        let mut steps = CHECK_STEPS;
        for (r, alts) in g.rules.iter().enumerate() {
            for a in 0..alts.len() { g.expand(vec![(r, a, 0)], 0, &mut steps, &mut vec![]); }
        }
        ensure!(steps > 0, "grammar: its expansion takes too many steps (left recursion, or alternatives that multiply)");
        Ok(g)
    }

    /// The stacks of the grammar's start.
    fn start(&self) -> Vec<Vec<(usize, usize, usize)>> {
        let (mut out, mut steps) = (vec![], CHECK_STEPS);
        for alt in 0..self.rules[self.root].len() { self.expand(vec![(self.root, alt, 0)], 0, &mut steps, &mut out); }
        out
    }

    /// The stacks `st` stands for with a character at the top of each (an empty stack: the grammar is complete).
    /// `steps` is the budget of calls left; at 0 the rest is dropped (`parse` refuses a grammar that gets there).
    fn expand(&self, mut st: Vec<(usize, usize, usize)>, depth: usize, steps: &mut usize, out: &mut Vec<Vec<(usize, usize, usize)>>) {
        // ponytail: a depth cut instead of rejecting left recursion when parsing.
        if depth > 200 || *steps == 0 { *steps = 0; return; }
        *steps -= 1;
        loop {
            let Some(&(r, a, e)) = st.last() else { out.push(st); return };
            match self.rules[r][a].get(e) {
                None => { st.pop(); }
                Some(Elem::Chars(..)) => { out.push(st); return; }
                Some(&Elem::Rule(k)) => {
                    st.last_mut().unwrap().2 += 1;
                    // A finished frame goes before the rule starts, so a repetition does not grow the stack.
                    while let Some(&(r, a, e)) = st.last() && e == self.rules[r][a].len() { st.pop(); }
                    for alt in 0..self.rules[k].len() {
                        let mut s = st.clone();
                        s.push((k, alt, 0));
                        self.expand(s, depth + 1, steps, out);
                    }
                    return;
                }
            }
        }
    }

    /// The stacks after the character `c`.
    fn advance(&self, stacks: &[Vec<(usize, usize, usize)>], c: u32) -> Vec<Vec<(usize, usize, usize)>> {
        let (mut out, mut steps) = (vec![], STEPS);
        for st in stacks {
            let Some(&(r, a, e)) = st.last() else { continue };
            if let Elem::Chars(neg, rs) = &self.rules[r][a][e] && *neg != rs.iter().any(|&(lo, hi)| (lo..=hi).contains(&c)) {
                let mut s = st.clone();
                s.last_mut().unwrap().2 += 1;
                self.expand(s, 0, &mut steps, &mut out);
            }
        }
        if out.len() > 1 { (out.sort_unstable(), out.dedup()); }
        out.truncate(MAX_STACKS);
        out
    }

    /// Whether some top could match a character from `lo` to `hi` (the start of a character split across tokens).
    fn may(&self, stacks: &[Vec<(usize, usize, usize)>], lo: u32, hi: u32) -> bool {
        stacks.iter().any(|st| st.last().is_some_and(|&(r, a, e)| match &self.rules[r][a][e] {
            Elem::Chars(false, rs) => rs.iter().any(|&(l, h)| l <= hi && h >= lo),
            Elem::Chars(true, rs) => !rs.iter().any(|&(l, h)| l <= lo && h >= hi),
            Elem::Rule(_) => false,
        }))
    }

    /// The stacks and the unfinished UTF-8 bytes after `bytes`, or None when the grammar does not allow them.
    fn feed(&self, stacks: &[Vec<(usize, usize, usize)>], part: &[u8], bytes: &[u8]) -> Option<(Vec<Vec<(usize, usize, usize)>>, Vec<u8>)> {
        let buf: std::borrow::Cow<[u8]> = if part.is_empty() { bytes.into() } else { [part, bytes].concat().into() };
        let mut cur = std::borrow::Cow::Borrowed(stacks);
        let mut i = 0;
        while i < buf.len() {
            let b = buf[i];
            let n = match b { 0..0x80 => 1, 0xc2..0xe0 => 2, 0xe0..0xf0 => 3, 0xf0..0xf5 => 4, _ => return None };
            let have = (buf.len() - i).min(n);
            let mut c = if n == 1 { b as u32 } else { (b & (0x7f >> n)) as u32 };
            for &x in &buf[i + 1..i + have] {
                if x & 0xc0 != 0x80 { return None; }
                c = c << 6 | (x & 0x3f) as u32;
            }
            // The smallest code point of each length: a longer (overlong) form, such as F0 80 81 BB for "{", is not
            // UTF-8, nor are surrogates.
            let least = [0, 0, 0x80, 0x800, 0x10000][n];
            if have < n {
                let k = 6 * (n - have) as u32;
                let (lo, hi) = ((c << k).max(least), ((c << k) | ((1 << k) - 1)).min(0x10ffff));
                return (lo <= hi && self.may(&cur, lo, hi)).then(|| (cur.into_owned(), buf[i..].to_vec()));
            }
            if c < least || c > 0x10ffff || (0xd800..0xe000).contains(&c) { return None; }
            cur = self.advance(&cur, c).into();
            if cur.is_empty() { return None; }
            i += n;
        }
        Some((cur.into_owned(), vec![]))
    }
}

/// Where a grammar is in the reply.
#[derive(Clone)]
struct Matcher { g: Arc<Grammar>, stacks: Vec<Vec<(usize, usize, usize)>>, part: Vec<u8>, bytes: Arc<Vec<Vec<u8>>> }

/// The `logit_bias` and `grammar` of one request.
#[derive(Clone)]
pub struct Constraint {
    bias: Vec<(u32, f32)>,
    gram: Option<Matcher>,
    /// The tokens that end the reply, which the grammar allows once it is complete.
    pub eos: Vec<u32>,
}

impl std::fmt::Debug for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { write!(f, "Constraint({} biases, grammar: {})", self.bias.len(), self.gram.is_some()) }
}

impl Constraint {
    /// None when the request has neither.
    pub fn new(q: &crate::serve::ChatRequest, tok: &crate::tokenizer::Tokenizer) -> anyhow::Result<Option<Constraint>> {
        Constraint::of(&q.logit_bias, q.grammar.clone(), tok)
    }

    pub fn of(bias: &[(BiasKey, f32)], grammar: Option<Arc<Grammar>>, tok: &crate::tokenizer::Tokenizer) -> anyhow::Result<Option<Constraint>> {
        if bias.is_empty() && grammar.is_none() { return Ok(None); }
        let mut b = vec![];
        for (k, v) in bias {
            match k {
                BiasKey::Id(id) => {
                    ensure!((*id as usize) < tok.vocab_size(), "logit_bias: {id} is not a token id");
                    b.push((*id, *v));
                }
                BiasKey::Text(s) => b.extend(tok.encode(s)?.into_iter().map(|id| (id, *v))),
            }
        }
        let gram = grammar.map(|g| {
            let stacks = g.start();
            Matcher { g, stacks, part: vec![], bytes: tok.grammar_bytes() }
        });
        Ok(Some(Constraint { bias: b, gram, eos: tok.eos_ids() }))
    }

    pub fn add_bias(&self, l: &mut [f32]) {
        for &(t, b) in &self.bias { if let Some(x) = l.get_mut(t as usize) { *x += b; } }
    }

    /// Whether the grammar allows token `t` next: an end-of-turn token only once the grammar is complete, a special
    /// token never.
    pub fn allows(&self, t: u32) -> bool {
        let Some(m) = &self.gram else { return true };
        if self.eos.contains(&t) { return m.part.is_empty() && m.stacks.iter().any(Vec::is_empty); }
        let b = m.bytes.get(t as usize).map_or(&[][..], Vec::as_slice);
        !b.is_empty() && m.g.feed(&m.stacks, &m.part, b).is_some()
    }

    /// Sets the logits of the tokens the grammar does not allow to -inf.
    /// ponytail: one walk of the whole vocabulary (about 10 ms for 150K tokens); a token trie if that matters.
    pub fn mask(&self, l: &mut [f32]) {
        if self.gram.is_none() { return; }
        for (t, x) in l.iter_mut().enumerate() {
            if *x != f32::NEG_INFINITY && !self.allows(t as u32) { *x = f32::NEG_INFINITY; }
        }
    }

    /// The bias, then the mask when the best token is not allowed (else it is not needed for a greedy pick).
    pub fn apply(&self, l: &mut [f32], greedy: bool) {
        self.add_bias(l);
        if !greedy || !self.allows(crate::ops::argmax(l) as u32) { self.mask(l); }
    }

    /// The reply got token `t`.
    pub fn accept(&mut self, t: u32) {
        let Some(m) = &mut self.gram else { return };
        if self.eos.contains(&t) { return; }
        let b = m.bytes.get(t as usize).map_or(&[][..], Vec::as_slice);
        (m.stacks, m.part) = m.g.feed(&m.stacks, &m.part, b).unwrap_or_default();
    }
}

/// The greedy choice of the separate model forwards: their argmax (the last of equal logits) unchanged without a
/// constraint, else after it, and the constraint follows the token.
pub fn greedy(lim: &mut Option<Constraint>, logits: &mut [f32]) -> u32 {
    let pick = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
    let Some(c) = lim else { return pick(logits) };
    c.add_bias(logits);
    let mut t = pick(logits);
    if !c.allows(t) { c.mask(logits); t = pick(logits); }
    c.accept(t);
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether the grammar takes `s` whole.
    fn full(g: &Grammar, s: &str) -> bool {
        let st = g.start();
        match g.feed(&st, &[], s.as_bytes()) { Some((st, part)) => part.is_empty() && st.iter().any(Vec::is_empty), None => false }
    }
    /// Whether `s` can start a text the grammar takes.
    fn prefix(g: &Grammar, s: &str) -> bool {
        let st = g.start();
        g.feed(&st, &[], s.as_bytes()).is_some()
    }

    // llama.cpp grammars/json.gbnf.
    const JSON: &str = r#"
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= | " " | "\n" [ \t]{0,20}
"#;

    // llama.cpp grammars/arithmetic.gbnf.
    const ARITH: &str = r#"
root  ::= (expr "=" ws term "\n")+
expr  ::= term ([-+*/] term)*
term  ::= ident | num | "(" ws expr ")" ws
ident ::= [a-z] [a-z0-9_]* ws
num   ::= [0-9]+ ws
ws    ::= [ \t\n]*
"#;

    // llama.cpp grammars/list.gbnf.
    const LIST: &str = r#"
root ::= item+

# Excludes various line break characters
item ::= "- " [^\r\n\x0b\x0c\x85  ]+ "\n"
"#;

    #[test]
    fn json_grammar() {
        let g = Grammar::parse(JSON).unwrap();
        assert!(full(&g, r#"{"a": [1, -2.5e3, true, null], "b": {"c": "x\"é ✓"}}"#));
        assert!(full(&g, "{}"));
        assert!(!full(&g, r#"{"a": 1,}"#));
        assert!(!full(&g, "[1]"));
        assert!(!full(&g, r#"{"a": 01x}"#));
        assert!(prefix(&g, r#"{"key": [1, 2"#));
        assert!(!prefix(&g, "hello"));
        // A character split across tokens: the start of "é" is allowed inside a string, not after "{".
        let st = g.start();
        let (s1, p) = g.feed(&st, &[], b"{\"").unwrap();
        assert!(p.is_empty());
        let (s2, p) = g.feed(&s1, &[], &[0xc3]).unwrap();
        assert_eq!(p, [0xc3]);
        assert!(g.feed(&s2, &p, &[0xa9]).is_some_and(|(_, p)| p.is_empty()));
        assert!(g.feed(&st, &[], b"{\xc3").is_none());
    }

    #[test]
    fn arithmetic_and_list_grammars() {
        let g = Grammar::parse(ARITH).unwrap();
        assert!(full(&g, "1+2=3\n"));
        assert!(full(&g, "(a*b_2) -7 = x\nx=1\n"));
        assert!(!full(&g, "(a * b) = 1\n"));
        assert!(!full(&g, "1+=3\n"));
        assert!(!full(&g, "1+2=3"));
        let g = Grammar::parse(LIST).unwrap();
        assert!(full(&g, "- one\n- two words\n"));
        assert!(!full(&g, "- \n"));
        assert!(!full(&g, "* one\n"));
    }

    #[test]
    fn grammar_syntax() {
        let g = Grammar::parse("root ::= \"a\"{2,3} [^x]? . # note\n").unwrap();
        assert!(full(&g, "aab") && full(&g, "aaa") && full(&g, "aaaqx") && full(&g, "aaaab") && !full(&g, "aaaxq") && !full(&g, "aaaaab") && !full(&g, "ab"));
        assert!(full(&Grammar::parse("root ::= (\"x\" | [0-9]){3,}").unwrap(), "x12x"));
        assert!(Grammar::parse("root ::= foo").is_err());
        assert!(Grammar::parse("start ::= \"a\"").is_err());
        assert!(Grammar::parse("root ::= \"a\"\nroot ::= \"b\"").is_err());
        assert!(Grammar::parse("root ::= (\"a\"").is_err());
        // Left recursion, and grammars that are too big to write out or expand, are refused (a request's grammar).
        assert!(Grammar::parse("root ::= root \"a\" | \"b\"").is_err());
        assert!(Grammar::parse(&format!("root ::= {}\"a\"{}", "(".repeat(100), ")".repeat(100))).is_err());
        assert!(Grammar::parse("root ::= \"a\"{100000}").is_err() && Grammar::parse("root ::= ((\"a\"{1000}){1000}){1000}").is_ok());
        let blow: String = (0..40).map(|i| format!("r{i} ::= r{} r{} | \"\"\n", i + 1, i + 1)).collect();
        assert!(Grammar::parse(&format!("root ::= r0\n{blow}r40 ::= \"y\"")).is_err());
        assert!(Grammar::parse(&"a".repeat(MAX_CHARS + 1)).is_err());
    }

    #[test]
    fn logit_bias_forms() {
        use serde_json::json;
        assert_eq!(parse_bias(&json!({"15": 5, "hi": -100})).unwrap(), [(BiasKey::Id(15), 5.0), (BiasKey::Text("hi".into()), f32::NEG_INFINITY)]);
        assert_eq!(parse_bias(&json!([[15, false], ["hi", 1.5]])).unwrap(), [(BiasKey::Id(15), f32::NEG_INFINITY), (BiasKey::Text("hi".into()), 1.5)]);
        assert!(parse_bias(&json!({"1": 101})).is_err() && parse_bias(&json!([[1]])).is_err() && parse_bias(&json!(3)).is_err());
    }
}
