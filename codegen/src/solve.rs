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
//! code. Vector code wants a different shape: conditions at consecutive `i`
//! that share one word gap, so that a window of them fits in one pair of
//! loads, and whose two bits one shift lines up. This module picks that basis
//! instead. It is the same span either way, so the check still matches
//! upstream's.
//!
//! What else the lanes of a group share is the target's [`Shape`]: the same
//! bit in every lane, or a bit of its own in each, which costs only a
//! different constant and leaves far more to choose from. The search runs
//! over slots, a window under one such key with the choices of each lane, and
//! fills a slot one lane at a time.
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
//! untried. A local search then swaps one or two units at a time while that
//! lowers the score, for about a tenth fewer entries at the same budget. A
//! unit may span two windows, since some groups are worth most together.
//!
//! Random data is not all a prefix sees. Every message ends in a block of
//! padding, where a condition passes or fails every time rather than half of
//! the time, and a prefix chosen for random blocks alone can leave several DVs
//! alive there. So for vector code both steps minimize [`objective`], which
//! also counts how often the last blocks of messages enter the tail and with
//! how many DVs.

use std::collections::{BTreeMap, HashMap};

use crate::emit::windows;
use crate::padding::{Alive, Exact, Last, Seen, half};
use crate::ubc::{UBCS, Ubc};

/// One bit of the expanded message.
type Vertex = (usize, u32);

/// One condition: bit `a` of `w[i]` XOR bit `b` of `w[j]` must equal `c`.
/// Failing it rules out the DVs in `dvs`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
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
///
/// A member may name its own near bit, when its target tests a bit per lane;
/// its far bit then keeps the family's distance from it, so one shift still
/// lines up every lane.
pub struct Family {
    pub offset: usize,
    pub near_bit: u32,
    pub far_bit: u32,
    /// Which value of the XOR bit clears the DV bits: 1 or 0.
    pub clears_on: u32,
    /// `(i, near bit, the DV bits that this check clears)`, in increasing
    /// order of `i`.
    pub members: Vec<(usize, u32, u32)>,
}

impl Family {
    /// The conditions the family checks, one per member.
    pub fn conds(&self) -> impl Iterator<Item = Cond> + '_ {
        let distance = self.far_bit as i32 - self.near_bit as i32;
        self.members.iter().map(move |&(i, a, dvs)| Cond {
            i,
            a,
            j: i + self.offset,
            b: (a as i32 + distance) as u32,
            c: self.clears_on ^ 1,
            dvs,
        })
    }
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

/// The expected number of DV bits that survive a prefix reaching `ranks`,
/// which is `sum 2^-r`. The scale is `2^-32` per unit, so that the rank of
/// the longest DV still lands on a whole number and all 32 fit in a `u64`.
fn survivors(ranks: &[u32; 32]) -> u64 {
    ranks.iter().map(|&r| 1u64 << (32 - r.min(32))).sum()
}

/// What the lanes of one vector group have in common.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Shape {
    /// One word gap, one bit pair and one polarity: every lane tests the same
    /// bit, against one constant.
    Strict,
    /// One word gap, one distance between the two bits and one polarity, so
    /// that a single shift still lines up every lane; each lane tests its own
    /// bit. Only the constant the lanes are tested against changes.
    LaneMask,
}

impl std::str::FromStr for Shape {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "strict" => Ok(Shape::Strict),
            "lanemask" => Ok(Shape::LaneMask),
            other => Err(format!("a shape is strict or lanemask, not {other}")),
        }
    }
}

/// What the lanes of a group must share under `shape`: the word gap, both
/// bits and the polarity for [`Shape::Strict`], or only the distance between
/// the bits in place of the bits for [`Shape::LaneMask`]. Keys sort in that
/// order, which is also the order ties are broken in.
fn key(shape: Shape, c: &Cond) -> (usize, i32, i32, u32) {
    match shape {
        Shape::Strict => (c.j - c.i, c.a as i32, c.b as i32, c.c),
        Shape::LaneMask => (c.j - c.i, c.b as i32 - c.a as i32, 0, c.c),
    }
}

/// Where a unit of the prefix can go: consecutive `i` under one key, and the
/// conditions each lane could check there. One window of `width` lanes is
/// one group; a unit may span [`WINDOWS`] of them, so that groups worth most
/// together can be chosen together.
type Slot = Vec<Vec<Cond>>;

/// How many windows one unit may span.
const WINDOWS: usize = 2;

/// How many groups the emitter makes of `unit`, by the same walk.
fn cost(unit: &[Cond], width: usize) -> usize {
    let is: Vec<usize> = unit.iter().map(|c| c.i).collect();
    windows(&is, width, unit[0].j - unit[0].i).len()
}

/// Every slot, from one window to [`WINDOWS`] of them. A slot of several
/// windows is only kept when its first and last windows can both be used,
/// since it would otherwise be a shorter slot again.
fn slots(cands: &[Cond], width: usize, shape: Shape) -> Vec<Slot> {
    let mut by: BTreeMap<(usize, i32, i32, u32), Vec<Cond>> = BTreeMap::new();
    for c in cands {
        by.entry(key(shape, c)).or_default().push(*c);
    }
    let mut out = Vec::new();
    for members in by.into_values() {
        let (lo, hi) = members
            .iter()
            .fold((usize::MAX, 0), |(lo, hi), c| (lo.min(c.i), hi.max(c.i)));
        let windows = if width > 1 { WINDOWS } else { 1 };
        for n in 1..=windows {
            let span = n * width;
            for base in lo.saturating_sub(span - 1)..=hi {
                let lanes: Slot = (0..span)
                    .map(|l| {
                        members
                            .iter()
                            .filter(|c| c.i == base + l)
                            .copied()
                            .collect()
                    })
                    .collect();
                let used = |w: usize| {
                    lanes[w * width..(w + 1) * width]
                        .iter()
                        .any(|l| !l.is_empty())
                };
                if used(0) && used(n - 1) {
                    out.push(lanes);
                }
            }
        }
    }
    out
}

/// Where a prefix stands: the union-find state and rank of every DV, and
/// where the last blocks of messages stand.
struct State {
    live: [Forest; 32],
    ranks: [u32; 32],
    last: Last,
}

impl State {
    /// Takes `fill` in, and gives back its group.
    fn commit(&mut self, fill: Fill) -> Vec<Cond> {
        for (dv, forest) in fill.trial {
            self.live[dv] = forest;
        }
        self.ranks = fill.ranks;
        for (d, (seen, alive)) in fill.last {
            self.last.put(d, seen, alive);
        }
        self.last.total();
        fill.group
    }
}

/// The DVs in `mask`.
pub(crate) fn dvs(mask: u32) -> impl Iterator<Item = usize> {
    (0..32).filter(move |d| mask >> d & 1 == 1)
}

/// A group filled against some state, and where it leaves that state: only
/// the DVs it changes, which is a few, rather than a copy of it all.
struct Fill {
    group: Vec<Cond>,
    trial: HashMap<usize, Forest>,
    ranks: [u32; 32],
    /// A map ordered by DV, so that the tail's products are taken in the
    /// same order on every run and the output does not depend on hashing.
    last: BTreeMap<usize, (Seen, Alive)>,
    score: f64,
}

/// The condition that does most for one lane so far, and what it changes.
struct Pick {
    score: f64,
    cond: Cond,
    ranks: [u32; 32],
    changed: Vec<(usize, Alive)>,
}

/// The prefix search for one target.
struct Search<'a> {
    bits: &'a Bits,
    width: usize,
    slots: Vec<Slot>,
    exact: Exact,
}

impl<'a> Search<'a> {
    fn new(bits: &'a Bits, cands: &[Cond], width: usize, shape: Shape) -> Self {
        Search {
            bits,
            width,
            slots: slots(cands, width, shape),
            // Vector prefixes are also scored on the last blocks of messages,
            // and ask for these candidates' sums there.
            exact: Exact::new(if width > 1 { cands } else { &[] }),
        }
    }

    /// Vector prefixes are scored by [`objective`]; the scalar one enters its
    /// tail almost always, and there the survivors alone are the measure.
    fn vector(&self) -> bool {
        self.width > 1
    }

    /// The score of a prefix reaching `ranks`, where the last blocks stand
    /// as in `last` but for the DVs in `changed`.
    fn score(&self, ranks: &[u32; 32], last: &Last, changed: &[(usize, &Alive)]) -> f64 {
        if self.vector() {
            objective(ranks, last.tail(changed))
        } else {
            survivors(ranks) as f64
        }
    }

    fn score_of(&self, state: &State) -> f64 {
        self.score(&state.ranks, &state.last, &[])
    }

    fn empty(&self) -> State {
        State {
            live: std::array::from_fn(|_| Forest::new(self.bits.vertex.len())),
            ranks: [0; 32],
            last: self.exact.empty(),
        }
    }

    /// Adds `group` to `state`, one condition at a time.
    fn add(&self, state: &mut State, group: &[Cond]) {
        for c in group {
            let (x, y) = (self.bits.index[&(c.i, c.a)], self.bits.index[&(c.j, c.b)]);
            for d in dvs(c.dvs) {
                if !state.live[d].union(x, y, c.c) {
                    continue;
                }
                state.ranks[d] += 1;
                if self.vector() {
                    self.exact.add(&mut state.last, d, c, state.ranks[d]);
                }
            }
        }
        state.last.total();
    }

    /// The state `units` reach from nothing.
    fn state(&self, units: &[Vec<Cond>]) -> State {
        let mut state = self.empty();
        for u in units {
            self.add(&mut state, u);
        }
        state
    }

    /// Fills `slot` against `state`, whose score is `score`, one lane at a
    /// time, each lane with the condition that lowers the score most given
    /// the lanes before it.
    fn fill(&self, state: &State, mut score: f64, slot: &Slot) -> Option<Fill> {
        let mut trial: HashMap<usize, Forest> = HashMap::new();
        let mut last: BTreeMap<usize, (Seen, Alive)> = BTreeMap::new();
        let mut ranks = state.ranks;
        let mut group = Vec::new();
        for options in slot {
            let mut pick: Option<Pick> = None;
            for c in options {
                // One condition joins two bits, so it adds one to the rank of
                // each of its DVs whose forest does not connect them yet.
                let (x, y) = (self.bits.index[&(c.i, c.a)], self.bits.index[&(c.j, c.b)]);
                let mut r = ranks;
                let mut gained = Vec::new();
                for d in dvs(c.dvs) {
                    let forest = trial.get(&d).unwrap_or(&state.live[d]);
                    if forest.find(x).0 != forest.find(y).0 {
                        r[d] += 1;
                        gained.push(d);
                    }
                }
                // A condition the others already imply adds no rank, and on
                // any block it fails only where they do: it clears nothing.
                if gained.is_empty() {
                    continue;
                }
                // The last blocks only add to the score, so an option whose
                // random part alone cannot win needs no more work.
                let bar = pick.as_ref().map_or(score, |p| p.score.min(score));
                if self.vector() && random(&r) >= bar {
                    continue;
                }
                let changed: Vec<(usize, Alive)> = if self.vector() {
                    gained
                        .iter()
                        .map(|&d| {
                            let seen = last.get(&d).map_or(&state.last.seen[d], |l| &l.0);
                            (d, self.exact.alive_after(seen, d, c, r[d]))
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                // The DVs this fill has changed already, and this condition's.
                let rows: Vec<(usize, &Alive)> = last
                    .iter()
                    .filter(|(d, _)| !changed.iter().any(|x| x.0 == **d))
                    .map(|(&d, l)| (d, &l.1))
                    .chain(changed.iter().map(|(d, a)| (*d, a)))
                    .collect();
                let s = self.score(&r, &state.last, &rows);
                if s < score && pick.as_ref().is_none_or(|p| s < p.score) {
                    pick = Some(Pick {
                        score: s,
                        cond: *c,
                        ranks: r,
                        changed,
                    });
                }
            }
            if let Some(p) = pick {
                apply(
                    &mut trial,
                    &state.live,
                    self.bits,
                    std::slice::from_ref(&p.cond),
                );
                (score, ranks) = (p.score, p.ranks);
                for (d, alive) in p.changed {
                    let seen = last.get(&d).map_or(&state.last.seen[d], |l| &l.0);
                    let seen = self.exact.seen_after(seen, d, &p.cond);
                    last.insert(d, (seen, alive));
                }
                group.push(p.cond);
            }
        }
        (!group.is_empty()).then_some(Fill {
            group,
            trial,
            ranks,
            last,
            score,
        })
    }

    /// The best unit to add to `state` within `free` groups: by the drop in
    /// score per group for the greedy, or by the score reached for the local
    /// search, which refills a budget already spent.
    fn best(&self, state: &State, free: usize, per_group: bool) -> Option<Fill> {
        let before = self.score_of(state);
        let mut best: Option<(f64, Fill)> = None;
        for slot in &self.slots {
            if slot.len() / self.width > free {
                continue;
            }
            let Some(f) = self.fill(state, before, slot) else {
                continue;
            };
            let groups = cost(&f.group, self.width);
            if groups > free {
                continue;
            }
            let rank = if per_group {
                -(before - f.score) / groups as f64
            } else {
                f.score
            };
            if best.as_ref().is_none_or(|b| rank < b.0) {
                best = Some((rank, f));
            }
        }
        best.map(|(_, f)| f)
    }

    fn spent(&self, units: &[Vec<Cond>]) -> usize {
        units.iter().map(|u| cost(u, self.width)).sum()
    }

    /// Adds the unit that pays most per group until `budget` groups are
    /// spent or none helps.
    fn greedy(&self, units: &mut Vec<Vec<Cond>>, budget: usize) -> State {
        let mut state = self.state(units);
        let mut spent = self.spent(units);
        while spent < budget {
            let Some(fill) = self.best(&state, budget - spent, true) else {
                break;
            };
            spent += cost(&fill.group, self.width);
            units.push(state.commit(fill));
        }
        state
    }

    /// Swaps one unit, or two, for what lowers the score most within the same
    /// budget, until no swap helps.
    fn refine(&self, units: &mut Vec<Vec<Cond>>, budget: usize) {
        let mut best = self.score_of(&self.state(units));
        loop {
            let mut improved = false;
            for k in 0..units.len() {
                let mut rest = units.clone();
                rest.remove(k);
                let free = budget - self.spent(&rest);
                if let Some(f) = self.best(&self.state(&rest), free, false)
                    && f.score < best - 1e-12
                {
                    best = f.score;
                    units[k] = f.group;
                    improved = true;
                }
            }
            let free = budget - self.spent(units);
            if free > 0
                && let Some(f) = self.best(&self.state(units), free, false)
                && f.score < best - 1e-12
            {
                best = f.score;
                units.push(f.group);
                improved = true;
            }
            if !improved {
                // Two out, and the budget they free refilled a unit at a
                // time, each the one that leaves the lowest score. Refilling
                // by the drop per group instead mostly undoes the move.
                'pairs: for i in 0..units.len() {
                    for j in i + 1..units.len() {
                        let mut rest: Vec<Vec<Cond>> = units
                            .iter()
                            .enumerate()
                            .filter(|&(n, _)| n != i && n != j)
                            .map(|(_, g)| g.clone())
                            .collect();
                        let mut state = self.state(&rest);
                        while let Some(f) = self.best(&state, budget - self.spent(&rest), false) {
                            rest.push(state.commit(f));
                        }
                        let s = self.score_of(&state);
                        if s < best - 1e-12 {
                            best = s;
                            *units = rest;
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
    }
}

/// Builds the check.
///
/// `width` is the lane count of a vector group, `groups` how many of them
/// the prefix may spend, and `shape` what the lanes of one group must share.
/// Everything the prefix leaves goes to the tail, where the cost is per
/// condition instead, so the two halves are chosen against different measures.
pub fn solve(width: usize, groups: usize, shape: Shape) -> Plan {
    let bits = Bits::collect();
    let goal = required();
    let cands = candidates(&bits, &spans(&bits));
    let search = Search::new(&bits, &cands, width, shape);

    // The greedy spends each group on what pays most at that moment, which
    // can leave a better set untried; a local search improves on it.
    let mut picked = Vec::new();
    let mut state = search.greedy(&mut picked, groups);
    if search.vector() && state.ranks.iter().sum::<u32>() < goal {
        search.refine(&mut picked, groups);
        state = search.state(&picked);
    }
    let State {
        mut live, ranks, ..
    } = state;
    let mut covered: u32 = ranks.iter().sum();

    let families: Vec<Family> = picked
        .iter()
        .map(|unit| {
            // A unit's members share the gap, the distance between their
            // bits and the polarity; the first names the shift.
            let first = unit[0];
            Family {
                offset: first.j - first.i,
                near_bit: first.a,
                far_bit: first.b,
                // `c` is the value the condition requires, so the other
                // value is the one that clears.
                clears_on: first.c ^ 1,
                members: unit.iter().map(|c| (c.i, c.a, c.dvs)).collect(),
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

/// How much the last block of a message counts against a block of random
/// data in [`objective`].
const PAD_WEIGHT: f64 = 0.02;

/// What the greedy and the local search minimize for vector code: how often
/// the tail runs and how many DVs it walks, on random blocks and on the last
/// blocks of messages.
fn objective(ranks: &[u32; 32], (enters, dvs): (f64, f64)) -> f64 {
    random(ranks) + PAD_WEIGHT * (enters + dvs)
}

/// The part of [`objective`] that random blocks make, which the last blocks
/// only add to.
fn random(ranks: &[u32; 32]) -> f64 {
    (1.0 - PAD_WEIGHT) * (p_tail(ranks) + survivors(ranks) as f64 / (1u64 << 32) as f64)
}

/// How often the tail runs on random data for a prefix reaching `ranks`.
fn p_tail(ranks: &[u32; 32]) -> f64 {
    1.0 - ranks.iter().map(|&r| 1.0 - half(r)).product::<f64>()
}
