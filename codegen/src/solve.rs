//! Chooses which conditions to check, and how to group them for vector code.
//!
//! # The freedom this exploits
//!
//! For one DV the published conditions are a basis of a linear space, not a
//! fixed list. Read them as a graph: a vertex is one bit of the expanded
//! message, and a condition is an edge that fixes the XOR of its two ends.
//! Inside a connected component the XOR of *any* two bits is then known, so
//! any spanning forest of the same components states the same thing. The
//! mask comes out identical.
//!
//! The published basis has the fewest distinct conditions, which suits scalar
//! code. Vector code wants a different shape: many conditions that share one
//! word gap and one bit pair, over a continuous range of `i`, so that eight
//! of them fit in one pair of loads. This module picks that basis instead.
//! It is the same span either way, so `matches_c_reference` still holds.

use std::collections::HashMap;

use crate::ubc::{UBCS, Ubc};

/// One bit of the expanded message.
type Vertex = (usize, u32);

/// One condition: bit `a` of `w[i]` XOR bit `b` of `w[j]` must equal `c`.
/// Failing it rules out the DVs in `dvs`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Cond {
    pub i: usize,
    pub a: u32,
    pub j: usize,
    pub b: u32,
    pub c: u32,
    pub dvs: u32,
}

/// What a vector form checks: bit `near_bit` of `w[i]` XOR bit `far_bit` of
/// `w[i + offset]`, for every `i` in `members`.
pub struct Family {
    pub offset: usize,
    pub near_bit: u32,
    pub far_bit: u32,
    /// Which value of the XOR bit clears the DV bits: 1 or 0.
    pub clears_on: u32,
    /// `(i, the DV bits that this check clears)`, in increasing order of `i`.
    pub members: Vec<(usize, u32)>,
}

/// The whole check: a vectorized prefix, then the rest one at a time.
pub struct Plan {
    pub families: Vec<Family>,
    pub tail: Vec<Cond>,
}

/// Union-find over message bits that also tracks the XOR of each bit with
/// its root. Two bits in one tree have a known XOR; two bits in different
/// trees do not.
#[derive(Clone)]
struct Forest {
    parent: Vec<u32>,
    parity: Vec<u32>,
    size: Vec<u32>,
}

impl Forest {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            parity: vec![0; n],
            size: vec![1; n],
        }
    }

    /// The root of `x`, and the XOR of `x` with that root.
    fn find(&self, mut x: u32) -> (u32, u32) {
        let mut p = 0;
        while self.parent[x as usize] != x {
            p ^= self.parity[x as usize];
            x = self.parent[x as usize];
        }
        (x, p)
    }

    /// Records that `u` XOR `v` is `c`. Returns false if that was already
    /// known, which makes the condition redundant for this DV.
    fn union(&mut self, u: u32, v: u32, c: u32) -> bool {
        let ((ru, pu), (rv, pv)) = (self.find(u), self.find(v));
        if ru == rv {
            return false;
        }
        let (a, b, p) = if self.size[ru as usize] < self.size[rv as usize] {
            (ru, rv, pu ^ pv ^ c)
        } else {
            (rv, ru, pu ^ pv ^ c)
        };
        self.parent[a as usize] = b;
        self.parity[a as usize] = p;
        self.size[b as usize] += self.size[a as usize];
        true
    }
}

/// The message bits the published conditions mention, in a dense numbering.
struct Bits {
    index: HashMap<Vertex, u32>,
    vertex: Vec<Vertex>,
}

impl Bits {
    fn collect() -> Self {
        let mut this = Self {
            index: HashMap::new(),
            vertex: Vec::new(),
        };
        for u in UBCS {
            this.get((u.i, u.a));
            this.get((u.j, u.b));
        }
        this
    }

    fn get(&mut self, v: Vertex) -> u32 {
        let next = self.vertex.len() as u32;
        *self.index.entry(v).or_insert_with(|| {
            self.vertex.push(v);
            next
        })
    }
}

/// Which DVs a condition holds for, which is every DV whose published
/// conditions already put both bits in one component with that XOR.
fn candidates(bits: &Bits, spans: &[Forest; 32]) -> Vec<Cond> {
    let mut found: HashMap<(u32, u32, u32), u32> = HashMap::new();
    for (dv, span) in spans.iter().enumerate() {
        // Group this DV's bits by component, keeping each bit's XOR with the
        // root so the XOR of any pair is a XOR of the two.
        let mut parts: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
        let mut seen = vec![false; bits.vertex.len()];
        for u in UBCS {
            if u.dvs >> dv & 1 == 0 {
                continue;
            }
            for v in [(u.i, u.a), (u.j, u.b)] {
                let v = bits.index[&v];
                if !std::mem::replace(&mut seen[v as usize], true) {
                    let (root, parity) = span.find(v);
                    parts.entry(root).or_default().push((v, parity));
                }
            }
        }
        for mut part in parts.into_values() {
            part.sort_by_key(|&(v, _)| bits.vertex[v as usize]);
            for (n, &(u, pu)) in part.iter().enumerate() {
                for &(v, pv) in &part[n + 1..] {
                    *found.entry((u, v, pu ^ pv)).or_default() |= 1 << dv;
                }
            }
        }
    }

    let mut out: Vec<Cond> = found
        .into_iter()
        .map(|((u, v, c), dvs)| {
            let ((i, a), (j, b)) = (bits.vertex[u as usize], bits.vertex[v as usize]);
            Cond { i, a, j, b, c, dvs }
        })
        .collect();
    out.sort_by_key(|c| (c.i, c.a, c.j, c.b, c.c));
    out
}

/// What the emitters group on: one word gap, one bit pair, one polarity.
type Signature = (usize, u32, u32, u32);

fn signature(c: &Cond) -> Signature {
    (c.j - c.i, c.a, c.b, c.c)
}

/// The per-DV union-find state after applying the published conditions,
/// which is the span every equivalent basis must reach.
fn spans(bits: &Bits) -> [Forest; 32] {
    let mut out = std::array::from_fn(|_| Forest::new(bits.vertex.len()));
    for u in UBCS {
        let (x, y) = (bits.index[&(u.i, u.a)], bits.index[&(u.j, u.b)]);
        for dv in 0..32 {
            if u.dvs >> dv & 1 == 1 {
                out[dv].union(x, y, u.c);
            }
        }
    }
    out
}

/// How many conditions the published basis states, counted once per DV. A
/// basis is equivalent when it reaches this same total.
fn required() -> u32 {
    UBCS.iter().map(|u: &Ubc| u.dvs.count_ones()).sum()
}

/// Applies `conds` to `state` and returns how many of the unions were new.
/// Only the DVs a condition names are touched, so a trial is cheap.
fn apply(
    state: &mut HashMap<usize, Forest>,
    base: &[Forest; 32],
    bits: &Bits,
    conds: &[Cond],
) -> u32 {
    let mut gain = 0;
    for c in conds {
        let (x, y) = (bits.index[&(c.i, c.a)], bits.index[&(c.j, c.b)]);
        for dv in 0..32 {
            if c.dvs >> dv & 1 == 0 {
                continue;
            }
            let forest = state.entry(dv).or_insert_with(|| base[dv].clone());
            if forest.union(x, y, c.c) {
                gain += 1;
            }
        }
    }
    gain
}

/// Builds the check.
///
/// `width` is the lane count the prefix is costed against, and `groups` is
/// how many vector groups it may spend. Everything left over goes to the
/// tail, where the cost is per condition instead, so the two halves are
/// chosen against different measures.
pub fn solve(width: usize, groups: usize) -> Plan {
    let bits = Bits::collect();
    let goal = required();
    let cands = candidates(&bits, &spans(&bits));

    // Continuous runs of `i` within one signature: what a vector group can
    // cover with a single pair of loads.
    let mut by_signature: HashMap<Signature, Vec<Cond>> = HashMap::new();
    for c in &cands {
        by_signature.entry(signature(c)).or_default().push(*c);
    }
    let mut runs: Vec<Vec<Cond>> = Vec::new();
    for mut group in by_signature.into_values() {
        group.sort_by_key(|c| c.i);
        let mut start = 0;
        for n in 1..=group.len() {
            if n == group.len() || group[n].i != group[n - 1].i + 1 {
                // Every sub-run is a candidate: a shorter one can be worth
                // more per group than the whole.
                for from in start..n {
                    for to in from + 1..=n {
                        runs.push(group[from..to].to_vec());
                    }
                }
                start = n;
            }
        }
    }

    // A HashMap decided the order above, and the greedy below keeps the
    // first of equal candidates, so sort to make the output reproducible.
    runs.sort_by_key(|r| (signature(&r[0]), r[0].i, r.len()));

    let mut live: [Forest; 32] = std::array::from_fn(|_| Forest::new(bits.vertex.len()));
    let mut covered = 0;
    let mut families: Vec<Family> = Vec::new();
    let mut spent = 0;

    while spent < groups && covered < goal {
        let mut best: Option<(u32, usize, Vec<Cond>, HashMap<usize, Forest>)> = None;
        for run in &runs {
            let cost = run.len().div_ceil(width);
            if spent + cost > groups {
                continue;
            }
            let mut trial = HashMap::new();
            let gain = apply(&mut trial, &live, &bits, run);
            if gain == 0 {
                continue;
            }
            // Per group, not per condition: a group is what costs.
            let better = match &best {
                None => true,
                Some((g, c, _, _)) => gain * *c as u32 > *g * cost as u32,
            };
            if better {
                best = Some((gain, cost, run.clone(), trial));
            }
        }
        let Some((gain, cost, run, trial)) = best else {
            break;
        };
        for (dv, forest) in trial {
            live[dv] = forest;
        }
        covered += gain;
        spent += cost;
        let s = signature(&run[0]);
        families.push(Family {
            offset: s.0,
            near_bit: s.1,
            far_bit: s.2,
            // `c` is the value the condition requires, so the other
            // value is the one that clears.
            clears_on: s.3 ^ 1,
            members: run.iter().map(|c| (c.i, c.dvs)).collect(),
        });
        runs.retain(|r| r != &run);
    }

    // The rest one at a time, where the measure is conditions rather than
    // groups, so this half wants the fewest of them.
    let mut tail = Vec::new();
    while covered < goal {
        let mut best: Option<(u32, Cond, HashMap<usize, Forest>)> = None;
        for c in &cands {
            let mut trial = HashMap::new();
            let gain = apply(&mut trial, &live, &bits, std::slice::from_ref(c));
            if gain > 0 && best.as_ref().is_none_or(|(g, _, _)| gain > *g) {
                best = Some((gain, *c, trial));
            }
        }
        let Some((gain, cond, trial)) = best else {
            break;
        };
        for (dv, forest) in trial {
            live[dv] = forest;
        }
        covered += gain;
        tail.push(cond);
    }

    assert_eq!(
        covered, goal,
        "the chosen basis does not span the published one"
    );
    Plan { families, tail }
}
