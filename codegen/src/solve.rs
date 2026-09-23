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
//!
//! # What the prefix is spent on
//!
//! The prefix runs on every block; the tail runs only when the prefix leaves
//! some DV bit set. A DV the prefix has covered to rank `r` leaves its bit
//! set with probability `2^-r`, so what the prefix buys is the drop in
//! [`survivors`], the expected number of survivors.
//!
//! Total coverage is the wrong measure for that. It values a DV's eleventh
//! condition the same as another DV's first, when the first is worth 1024
//! times more, and a greedy spending on coverage will leave a DV at rank 0
//! while it deepens one already at rank 11. That leaves the tail entered on
//! every block, and the `mask == 0` return after the prefix unreachable.
//!
//! A greedy commits to each group as it goes, which can leave a better set
//! untried. A local search then swaps one or two groups at a time while that
//! lowers how often the tail runs at all, for about a tenth fewer entries at
//! the same budget.

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
    /// What rank the prefix reaches for each DV. A DV at rank `r` survives
    /// the prefix, and so enters the tail, with probability `2^-r`.
    pub prefix_ranks: [u32; 32],
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
        for (dv, forest) in out.iter_mut().enumerate() {
            if u.dvs >> dv & 1 == 1 {
                forest.union(x, y, u.c);
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

/// A candidate step: what it saves in expected survivors, what it costs in
/// groups, the conditions it adds, the union-find state that results, and
/// the rank it gains for each DV.
type Candidate = (u64, usize, Vec<Cond>, HashMap<usize, Forest>, [u32; 32]);

/// Applies `conds` to `state` and returns how many of the unions were new,
/// per DV. Only the DVs a condition names are touched, so a trial is cheap.
fn apply(
    state: &mut HashMap<usize, Forest>,
    base: &[Forest; 32],
    bits: &Bits,
    conds: &[Cond],
) -> [u32; 32] {
    let mut gain = [0; 32];
    for c in conds {
        let (x, y) = (bits.index[&(c.i, c.a)], bits.index[&(c.j, c.b)]);
        for (dv, start) in base.iter().enumerate() {
            if c.dvs >> dv & 1 == 0 {
                continue;
            }
            let forest = state.entry(dv).or_insert_with(|| start.clone());
            if forest.union(x, y, c.c) {
                gain[dv] += 1;
            }
        }
    }
    gain
}

/// How many groups a run takes, which is how many pairs of loads the
/// emitter will make of it.
///
/// One pair reaches `width` consecutive `i`, so a group takes every member
/// that falls in that window and the next one starts over. This is the same
/// walk `emit::lane_groups` does, so the budget the solver spends is the
/// number of groups that come out.
fn groups_for(run: &[Cond], width: usize) -> usize {
    let mut groups = 0;
    let mut rest = run;
    while let Some(first) = rest.first() {
        let end = rest.partition_point(|c| c.i < first.i + width);
        rest = &rest[end..];
        groups += 1;
    }
    groups
}

/// The expected number of DV bits that survive a prefix reaching `ranks`,
/// which is `sum 2^-r`. The scale is `2^-32` per unit, so that the rank of
/// the longest DV still lands on a whole number and all 32 fit in a `u64`.
fn survivors(ranks: &[u32; 32]) -> u64 {
    ranks.iter().map(|&r| 1u64 << (32 - r.min(32))).sum()
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
    let all = runs_of(&cands);
    let mut runs = all.clone();

    let mut live: [Forest; 32] = std::array::from_fn(|_| Forest::new(bits.vertex.len()));
    let mut ranks = [0u32; 32];
    let mut covered = 0;
    let mut picked: Vec<Vec<Cond>> = Vec::new();
    let mut spent = 0;

    while spent < groups && covered < goal {
        let mut best: Option<Candidate> = None;
        let before = survivors(&ranks);
        for run in &runs {
            let cost = groups_for(run, width);
            if spent + cost > groups {
                continue;
            }
            let mut trial = HashMap::new();
            let gain = apply(&mut trial, &live, &bits, run);
            if gain.iter().sum::<u32>() == 0 {
                continue;
            }
            let mut after = ranks;
            for dv in 0..32 {
                after[dv] += gain[dv];
            }
            // What the group buys is the drop in expected survivors, and
            // what it costs is one group, not one condition.
            let drop = before - survivors(&after);
            let better = match &best {
                None => true,
                Some((d, c, _, _, _)) => drop * *c as u64 > *d * cost as u64,
            };
            if better {
                best = Some((drop, cost, run.clone(), trial, gain));
            }
        }
        let Some((_, cost, run, trial, gain)) = best else {
            break;
        };
        for (dv, forest) in trial {
            live[dv] = forest;
        }
        for dv in 0..32 {
            ranks[dv] += gain[dv];
        }
        covered += gain.iter().sum::<u32>();
        spent += cost;
        runs.retain(|r| r != &run);
        picked.push(run);
    }

    // The greedy spends each group on what pays most at that moment, which
    // can leave a better set untried; a local search improves on it. It
    // minimizes how often the tail runs, which is what a vector prefix is
    // for; the scalar one enters its tail almost always, and there the
    // greedy's own measure, the survivors, is the right one.
    if covered < goal && width > 1 {
        picked = refine(&bits, &all, picked, width, groups);
        let chosen: Vec<&Vec<Cond>> = picked.iter().collect();
        (live, ranks) = reach(&bits, &chosen);
        covered = ranks.iter().sum();
    }
    let families: Vec<Family> = picked
        .iter()
        .map(|run| {
            let s = signature(&run[0]);
            Family {
                offset: s.0,
                near_bit: s.1,
                far_bit: s.2,
                // `c` is the value the condition requires, so the other
                // value is the one that clears.
                clears_on: s.3 ^ 1,
                members: run.iter().map(|c| (c.i, c.dvs)).collect(),
            }
        })
        .collect();

    // The rest one at a time, where the measure is conditions rather than
    // groups, so this half wants the fewest of them.
    let mut tail = Vec::new();
    while covered < goal {
        let mut best: Option<(u32, Cond, HashMap<usize, Forest>)> = None;
        for c in &cands {
            let mut trial = HashMap::new();
            let gain: u32 = apply(&mut trial, &live, &bits, std::slice::from_ref(c))
                .iter()
                .sum();
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
    Plan {
        families,
        tail,
        prefix_ranks: ranks,
    }
}

/// How often the tail runs on random data for a prefix reaching `ranks`.
fn p_tail(ranks: &[u32; 32]) -> f64 {
    1.0 - ranks
        .iter()
        .map(|&r| 1.0 - 0.5f64.powi(r as i32))
        .product::<f64>()
}

/// Continuous runs of `i` within one signature: what a vector group can
/// cover with a single pair of loads.
///
/// Every run of conditions in one signature is a candidate. A shorter run
/// can be worth more per group than the whole, and a run may step over an
/// `i` with no condition: the lane it leaves empty takes no DV bits and
/// clears nothing. What a hole costs is the lanes it wastes, which
/// [`groups_for`] prices and the solver weighs.
fn runs_of(cands: &[Cond]) -> Vec<Vec<Cond>> {
    let mut by_signature: HashMap<Signature, Vec<Cond>> = HashMap::new();
    for c in cands {
        by_signature.entry(signature(c)).or_default().push(*c);
    }
    let mut runs: Vec<Vec<Cond>> = Vec::new();
    for mut group in by_signature.into_values() {
        group.sort_by_key(|c| c.i);
        for from in 0..group.len() {
            for to in from + 1..=group.len() {
                runs.push(group[from..to].to_vec());
            }
        }
    }
    // A HashMap decided the order above, and the solver keeps the first of
    // equal candidates, so sort to make the output reproducible.
    runs.sort_by_key(|r| (signature(&r[0]), r[0].i, r.len()));
    runs
}

/// The forests and ranks a set of runs reaches.
fn reach(bits: &Bits, chosen: &[&Vec<Cond>]) -> ([Forest; 32], [u32; 32]) {
    let mut forests: [Forest; 32] = std::array::from_fn(|_| Forest::new(bits.vertex.len()));
    let mut ranks = [0; 32];
    for run in chosen {
        for c in run.iter() {
            let (x, y) = (bits.index[&(c.i, c.a)], bits.index[&(c.j, c.b)]);
            for (dv, f) in forests.iter_mut().enumerate() {
                if c.dvs >> dv & 1 == 1 && f.union(x, y, c.c) {
                    ranks[dv] += 1;
                }
            }
        }
    }
    (forests, ranks)
}

/// The best run to add to `chosen` within `free` groups, if it beats `best`.
fn best_addition(
    bits: &Bits,
    runs: &[Vec<Cond>],
    chosen: &[usize],
    free: usize,
    width: usize,
    best: f64,
) -> Option<(f64, usize)> {
    let (forests, ranks) = reach(bits, &chosen.iter().map(|&k| &runs[k]).collect::<Vec<_>>());
    let mut pick: Option<(f64, usize)> = None;
    for (k, run) in runs.iter().enumerate() {
        if groups_for(run, width) > free || chosen.contains(&k) {
            continue;
        }
        let gain = apply(&mut HashMap::new(), &forests, bits, run);
        let mut r = ranks;
        for dv in 0..32 {
            r[dv] += gain[dv];
        }
        let p = p_tail(&r);
        if p < best - 1e-12 && pick.is_none_or(|(q, _)| p < q) {
            pick = Some((p, k));
        }
    }
    pick
}

/// Improves a prefix by local search: swap one run, or two, for the runs that
/// lower P(tail) most within the same budget, until no swap helps.
fn refine(
    bits: &Bits,
    runs: &[Vec<Cond>],
    picked: Vec<Vec<Cond>>,
    width: usize,
    groups: usize,
) -> Vec<Vec<Cond>> {
    let mut chosen: Vec<usize> = picked
        .iter()
        .map(|p| {
            runs.iter()
                .position(|r| r == p)
                .expect("a greedy run not among the candidates")
        })
        .collect();
    let cost = |ix: &[usize]| {
        ix.iter()
            .map(|&k| groups_for(&runs[k], width))
            .sum::<usize>()
    };
    let p_of =
        |ix: &[usize]| p_tail(&reach(bits, &ix.iter().map(|&k| &runs[k]).collect::<Vec<_>>()).1);
    let mut best = p_of(&chosen);
    loop {
        let mut improved = false;
        for slot in 0..chosen.len() {
            let mut rest = chosen.clone();
            rest.remove(slot);
            let free = groups - cost(&rest);
            if let Some((p, k)) = best_addition(bits, runs, &rest, free, width, best) {
                chosen[slot] = k;
                best = p;
                improved = true;
            }
        }
        let free = groups - cost(&chosen);
        if free > 0
            && let Some((p, k)) = best_addition(bits, runs, &chosen, free, width, best)
        {
            chosen.push(k);
            best = p;
            improved = true;
        }
        // Two out, the best two back in, one at a time.
        if !improved {
            'pairs: for i in 0..chosen.len() {
                for j in i + 1..chosen.len() {
                    let mut rest: Vec<usize> = chosen
                        .iter()
                        .enumerate()
                        .filter(|&(n, _)| n != i && n != j)
                        .map(|(_, &k)| k)
                        .collect();
                    for _ in 0..2 {
                        let free = groups - cost(&rest);
                        if let Some((_, k)) = best_addition(bits, runs, &rest, free, width, 1.0) {
                            rest.push(k);
                        }
                    }
                    let p = p_of(&rest);
                    if p < best - 1e-12 {
                        chosen = rest;
                        best = p;
                        improved = true;
                        break 'pairs;
                    }
                }
            }
        }
        if !improved {
            break;
        }
    }
    chosen.into_iter().map(|k| runs[k].clone()).collect()
}
