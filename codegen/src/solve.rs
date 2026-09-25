//! Chooses which conditions to check, and how to group them for vector code.
//!
//! # The freedom this exploits
//!
//! Per DV the published conditions are a basis, not a fixed list: as a
//! graph of message bits joined by known XORs, any spanning forest of the
//! same components gives the same mask. Vector code wants conditions at
//! consecutive `i` that share a word gap, so that one pair of loads covers a
//! window of them, and that one shift lines up. This module picks such a
//! basis; the span, and so the check, stays upstream's.
//!
//! What else a group's lanes share depends on the target's [`Ops`]. The
//! search fills slots, a window of lanes under one key, one lane at a time.
//!
//! # What the prefix is spent on
//!
//! The tail runs only when the prefix leaves a DV bit set, which a DV at
//! rank `r` does with probability `2^-r`. So the prefix buys a drop in
//! [`survivors`], not coverage: a DV's first condition is worth 1024 times
//! its eleventh.
//!
//! A variable neighbourhood search improves on the greedy. A unit may span
//! two windows, since some groups are worth most together.
//!
//! A message's last block is mostly padding, where a condition passes or
//! fails every time, so a vector prefix is scored by [`objective`], which
//! counts those blocks too.

use std::collections::{BTreeMap, HashMap};

use rand::rngs::Xoshiro256PlusPlus;
use rand::{RngExt as _, SeedableRng as _};

use crate::padding::{Alive, Exact, Last, Seen, half, mean};
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
/// With a bit per lane, a member names its own near bit, at the family's
/// distance from its far bit.
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
    /// The family a unit of the prefix makes. Its members share the gap, the
    /// distance between their bits and the polarity; the first names the
    /// shift.
    fn of(unit: &[Cond]) -> Family {
        let first = unit[0];
        Family {
            offset: first.j - first.i,
            near_bit: first.a,
            far_bit: first.b,
            // `c` is the value the condition requires, so the other value is
            // the one that clears.
            clears_on: first.c ^ 1,
            members: unit.iter().map(|c| (c.i, c.a, c.dvs)).collect(),
        }
    }

    /// The conditions the family checks, one per member.
    pub(crate) fn conds(&self) -> impl Iterator<Item = Cond> + '_ {
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
    /// Each bit's number, by `32 * word + bit`, or `u32::MAX` for a bit no
    /// condition mentions. A table, for the search's innermost loops.
    index: Vec<u32>,
    vertex: Vec<Vertex>,
}

impl Bits {
    fn collect() -> Self {
        let mut this = Self {
            index: vec![u32::MAX; 80 * 32],
            vertex: Vec::new(),
        };
        for u in UBCS {
            this.get((u.i, u.a));
            this.get((u.j, u.b));
        }
        this
    }

    fn get(&mut self, v: Vertex) {
        let slot = &mut self.index[32 * v.0 + v.1 as usize];
        if *slot == u32::MAX {
            *slot = self.vertex.len() as u32;
            self.vertex.push(v);
        }
    }

    /// The number of bit `v`.
    fn id(&self, (i, a): Vertex) -> u32 {
        let n = self.index[32 * i + a as usize];
        debug_assert_ne!(n, u32::MAX, "w[{i}] bit {a} is in no condition");
        n
    }

    /// The numbers of the two bits `c` joins.
    fn ends(&self, c: &Cond) -> (u32, u32) {
        (self.id((c.i, c.a)), self.id((c.j, c.b)))
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
                let v = bits.id(v);
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
        let (x, y) = (bits.id((u.i, u.a)), bits.id((u.j, u.b)));
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

/// The forests of the DVs a trial has changed, the others still being those
/// it started from.
type Trial = [Option<Forest>; 32];

/// Applies `c` to `trial`, whose forests start as `base`. Only the DVs `c`
/// names are copied, so a trial is cheap.
fn apply(trial: &mut Trial, base: &[Forest; 32], bits: &Bits, c: &Cond) {
    let (x, y) = bits.ends(c);
    for d in dvs(c.dvs) {
        trial[d]
            .get_or_insert_with(|| base[d].clone())
            .union(x, y, c.c);
    }
}

/// The expected number of DV bits that survive a prefix reaching `ranks`,
/// which is `sum 2^-r`. The scale is `2^-32` per unit, so that the rank of
/// the longest DV still lands on a whole number and all 32 fit in a `u64`.
fn survivors(ranks: &[u32; 32]) -> u64 {
    ranks.iter().map(|&r| 1u64 << (32 - r.min(32))).sum()
}

/// What a target can do for the lanes of one group, and so what they may
/// differ in. They always share the word gap and the polarity.
#[derive(Clone, Copy)]
pub struct Ops {
    /// Test each lane against its own bit, so that lanes share only the
    /// distance between their two bits.
    pub lane_bit: bool,
}

/// What the lanes of a group must share under `ops`.
fn key(ops: Ops, c: &Cond) -> (usize, i32, i32, u32) {
    let (gap, distance) = (c.j - c.i, c.b as i32 - c.a as i32);
    if ops.lane_bit {
        (gap, distance, 0, c.c)
    } else {
        (gap, c.a as i32, c.b as i32, c.c)
    }
}

/// Where a unit can go: consecutive `i` under one key, with what each lane
/// could check. A window of `width` lanes is one group.
type Slot = Vec<Vec<Cond>>;

/// How many windows one unit may span.
const WINDOWS: usize = 2;

/// Words in the expanded message schedule. Every emitted form takes
/// `&[u32; SCHEDULE]`, and a read past it is out of bounds.
pub const SCHEDULE: usize = 80;

/// Where the groups of a family go: the start of each window of `width`
/// lanes, one pair of loads, and how many of the members `is` it takes. A
/// window starts at its first member, or earlier if its far load would run
/// past the schedule. The budget is counted with this same walk.
pub fn windows(is: &[usize], width: usize, offset: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut rest = is;
    while let Some(&first) = rest.first() {
        let base = first.min(SCHEDULE - width - offset);
        let n = rest.iter().take_while(|&&i| i < base + width).count();
        out.push((base, n));
        rest = &rest[n..];
    }
    out
}

/// How many groups the emitter makes of `unit`, by the same walk.
fn cost(unit: &[Cond], width: usize) -> usize {
    let is: Vec<usize> = unit.iter().map(|c| c.i).collect();
    windows(&is, width, unit[0].j - unit[0].i).len()
}

/// Every slot, from one window to [`WINDOWS`] of them, each using its
/// first and last windows.
fn slots(cands: &[Cond], width: usize, ops: Ops) -> Vec<Slot> {
    let mut by: BTreeMap<(usize, i32, i32, u32), Vec<Cond>> = BTreeMap::new();
    for c in cands {
        by.entry(key(ops, c)).or_default().push(*c);
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
        for (d, forest) in fill.trial.into_iter().enumerate() {
            if let Some(forest) = forest {
                self.live[d] = forest;
            }
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
pub(crate) fn dvs(mut mask: u32) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        (mask != 0).then(|| {
            let d = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            d
        })
    })
}

/// A group filled against some state, and where it leaves that state: only
/// the DVs it changes, which is a few, rather than a copy of it all.
struct Fill {
    group: Vec<Cond>,
    trial: Trial,
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

/// What every fill in one [`Search::best`] starts from.
struct Frame {
    before: f64,
    per_group: bool,
    roots: Vec<u32>,
    means: [f64; 32],
}

/// For each lane of a slot on, the ranks the rest could add per DV, and the
/// DVs they could touch.
type Reach = Vec<([u8; 32], u32)>;

/// How [`Search::best`] ranks a fill; lower is better.
fn rank(frame: &Frame, score: f64, groups: usize) -> f64 {
    if frame.per_group {
        -(frame.before - score) / groups as f64
    } else {
        score
    }
}

/// The prefix search for one target.
struct Search {
    bits: Bits,
    /// Every condition some DV's span holds.
    cands: Vec<Cond>,
    width: usize,
    slots: Vec<Slot>,
    exact: Exact,
}

impl Search {
    fn new(width: usize, ops: Ops) -> Self {
        let bits = Bits::collect();
        let spans = spans(&bits);
        let cands = candidates(&bits, &spans);
        Search {
            slots: slots(&cands, width, ops),
            // Vector prefixes are also scored on the last blocks of messages,
            // and ask for these candidates' sums there.
            exact: Exact::new(if width > 1 { &cands } else { &[] }),
            bits,
            cands,
            width,
        }
    }

    /// Whether this is a vector prefix.
    fn vector(&self) -> bool {
        self.width > 1
    }

    /// The score of a prefix at `state`: [`objective`] for a vector prefix,
    /// and the survivors alone for the scalar one, which measured about 3%
    /// slower from 65 bytes up when scored by [`objective`].
    fn score_of(&self, state: &State) -> f64 {
        if self.vector() {
            objective(&state.ranks, state.last.tail())
        } else {
            survivors(&state.ranks) as f64
        }
    }

    /// Adds `group` to `state`, one condition at a time.
    fn add(&self, state: &mut State, group: &[Cond]) {
        for c in group {
            let (x, y) = self.bits.ends(c);
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
        let mut state = State {
            live: std::array::from_fn(|_| Forest::new(self.bits.vertex.len())),
            ranks: [0; 32],
            last: self.exact.empty(),
        };
        for u in units {
            self.add(&mut state, u);
        }
        state
    }

    /// What each lane of `slot` on could still add, given `roots`.
    fn reach(&self, roots: &[u32], slot: &Slot) -> Reach {
        let mut reach = vec![([0u8; 32], 0u32); slot.len() + 1];
        for (k, options) in slot.iter().enumerate().rev() {
            let hit = self.hits(roots, options);
            let (mut adds, touched) = reach[k + 1];
            for d in dvs(hit) {
                adds[d] += 1;
            }
            reach[k] = (adds, touched | hit);
        }
        reach
    }

    /// The first entry of [`Search::reach`], alone.
    fn total(&self, roots: &[u32], slot: &Slot) -> ([u8; 32], u32) {
        slot.iter()
            .fold(([0u8; 32], 0), |(mut adds, touched), options| {
                let hit = self.hits(roots, options);
                for d in dvs(hit) {
                    adds[d] += 1;
                }
                (adds, touched | hit)
            })
    }

    /// The DVs some option of one lane would add rank to.
    fn hits(&self, roots: &[u32], options: &[Cond]) -> u32 {
        let n = self.bits.vertex.len();
        let mut hit = 0u32;
        for c in options {
            let (x, y) = self.bits.ends(c);
            for d in dvs(c.dvs & !hit) {
                if roots[d * n + x as usize] != roots[d * n + y as usize] {
                    hit |= 1 << d;
                }
            }
        }
        hit
    }

    /// A score no fill can go below from `ranks`, if the lanes left add all
    /// of `reach`. A DV they cannot touch keeps its last-block chance in
    /// `means`, so last blocks carry at least those DVs in.
    fn low(&self, ranks: &[u32; 32], (adds, touched): &([u8; 32], u32), means: &[f64; 32]) -> f64 {
        let ranks: [u32; 32] = std::array::from_fn(|d| ranks[d] + u32::from(adds[d]));
        if self.vector() {
            let kept = dvs(!touched).map(|d| means[d]);
            let (enters, carried) = kept.fold((0.0, 0.0), |(e, c), m: f64| (m.max(e), c + m));
            // Less a margin for rounding in another order.
            objective(&ranks, (enters, carried)) - 1e-12
        } else {
            survivors(&ranks) as f64
        }
    }

    /// Fills `slot` against `state` one lane at a time, each with the
    /// condition that lowers the score most, and gives up once the fill
    /// cannot rank below `cut`.
    fn fill(
        &self,
        state: &State,
        frame: &Frame,
        reach: &Reach,
        cut: f64,
        slot: &Slot,
    ) -> Option<Fill> {
        let mut trial: Trial = std::array::from_fn(|_| None);
        let mut last: BTreeMap<usize, (Seen, Alive)> = BTreeMap::new();
        let mut means = frame.means;
        // Kept as lanes are picked, so an option costs only its own DVs.
        let mut totals = state.last.totals().clone();
        let mut ranks = state.ranks;
        let mut score = frame.before;
        let mut group = Vec::new();
        for (k, options) in slot.iter().enumerate() {
            if k > 0 && cut.is_finite() {
                let low = self.low(&ranks, &reach[k], &means);
                let spent = if group.is_empty() {
                    1
                } else {
                    cost(&group, self.width)
                };
                if low >= frame.before || rank(frame, low, spent) > cut {
                    return None;
                }
            }
            let mut pick: Option<Pick> = None;
            for c in options {
                // One condition joins two bits, so it adds one to the rank of
                // each of its DVs whose forest does not connect them yet.
                let (x, y) = self.bits.ends(c);
                let mut r = ranks;
                let mut gained = Vec::new();
                for d in dvs(c.dvs) {
                    let forest = trial[d].as_ref().unwrap_or(&state.live[d]);
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
                let (changed, s) = if self.vector() {
                    if random(&r) >= bar {
                        continue;
                    }
                    let changed: Vec<(usize, Alive)> = gained
                        .iter()
                        .map(|&d| (d, self.exact.alive_after(at(&last, state, d).0, d, c, r[d])))
                        .collect();
                    let pairs: Vec<(&Alive, &Alive)> = changed
                        .iter()
                        .map(|(d, a)| (at(&last, state, *d).1, a))
                        .collect();
                    let s = objective(&r, totals.tail(pairs.iter().copied()));
                    (changed, s)
                } else {
                    (Vec::new(), survivors(&r) as f64)
                };
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
                apply(&mut trial, &state.live, &self.bits, &p.cond);
                (score, ranks) = (p.score, p.ranks);
                for (d, alive) in p.changed {
                    let (seen, was) = at(&last, state, d);
                    totals.swap(was, &alive);
                    let seen = self.exact.seen_after(seen, d, &p.cond);
                    means[d] = mean(&alive);
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
    /// score per group when `per_group`, and by the score reached otherwise.
    fn best(&self, state: &State, free: usize, per_group: bool) -> Option<Fill> {
        let n = self.bits.vertex.len();
        let mut roots = vec![0u32; 32 * n];
        for (d, forest) in state.live.iter().enumerate() {
            for v in 0..n {
                roots[d * n + v] = forest.find(v as u32).0;
            }
        }
        let frame = Frame {
            before: self.score_of(state),
            per_group,
            roots,
            means: state.last.means(),
        };
        // Filled most promising first, until no bound can win.
        let mut order: Vec<(f64, usize)> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.len() / self.width <= free)
            .map(|(n, slot)| {
                let total = self.total(&frame.roots, slot);
                let bound = self.low(&state.ranks, &total, &frame.means);
                let key = if per_group {
                    bound - frame.before
                } else {
                    bound
                };
                (key, n)
            })
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        // The lowest rank, and among equals the first slot.
        let mut best: Option<(f64, usize, Fill)> = None;
        for (bound, n) in order {
            let cut = best.as_ref().map_or(f64::INFINITY, |b| b.0);
            if bound > cut {
                break;
            }
            let slot = &self.slots[n];
            let reach = self.reach(&frame.roots, slot);
            let Some(f) = self.fill(state, &frame, &reach, cut, slot) else {
                continue;
            };
            let groups = cost(&f.group, self.width);
            if groups > free {
                continue;
            }
            let rank = rank(&frame, f.score, groups);
            if best
                .as_ref()
                .is_none_or(|b| rank < b.0 || (rank == b.0 && n < b.1))
            {
                best = Some((rank, n, f));
            }
        }
        best.map(|(_, _, f)| f)
    }

    fn spent(&self, units: &[Vec<Cond>]) -> usize {
        units.iter().map(|u| cost(u, self.width)).sum()
    }

    /// Adds what [`Search::best`] picks while it fits `budget`, and gives back
    /// the state. With `per_group` this is the greedy.
    fn refill(&self, units: &mut Vec<Vec<Cond>>, budget: usize, per_group: bool) -> State {
        let mut state = self.state(units);
        while let Some(f) = self.best(&state, budget - self.spent(units), per_group) {
            units.push(state.commit(f));
        }
        state
    }

    /// Takes `k` of `units` out at random and refills the budget they free.
    fn shake(
        &self,
        units: &[Vec<Cond>],
        budget: usize,
        k: usize,
        rng: &mut Xoshiro256PlusPlus,
    ) -> (Vec<Vec<Cond>>, State) {
        let mut out: Vec<usize> = (0..units.len()).collect();
        for n in 0..k {
            let m = rng.random_range(n..out.len());
            out.swap(n, m);
        }
        out.truncate(k);
        let mut rest = without(units, &out);
        let per_group = rng.random_bool(PER_GROUP);
        let state = self.refill(&mut rest, budget, per_group);
        (rest, state)
    }

    /// Shakes up to [`STEP`] units at a time, keeping any prefix no worse,
    /// until [`DESCENT`] rounds pass without a better one. Leaves the best in
    /// `units` and gives back its score.
    fn descend(
        &self,
        units: &mut Vec<Vec<Cond>>,
        budget: usize,
        rng: &mut Xoshiro256PlusPlus,
    ) -> f64 {
        let mut current = units.clone();
        let mut score = self.score_of(&self.state(&current));
        let mut best = score;
        let mut found = 0;
        for round in 0.. {
            if current.is_empty() || round - found > DESCENT {
                break;
            }
            let k = rng.random_range(1..=STEP.min(current.len()));
            let (rest, state) = self.shake(&current, budget, k, rng);
            let s = self.score_of(&state);
            if s <= score {
                current = rest;
                score = s;
                if s < best - 1e-12 {
                    best = s;
                    found = round;
                    *units = current.clone();
                }
            }
        }
        best
    }

    /// Variable neighbourhood search: descends, then shakes `k` units beyond a
    /// descent's reach and descends again, keeping only a better prefix. `k`
    /// resets on success and grows to [`SHAKE`] otherwise, until more than
    /// `stall` shakes fail in a row. The generator is portable, so every
    /// machine finds the same prefix.
    fn vns(&self, units: &mut Vec<Vec<Cond>>, budget: usize, stall: usize) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let mut best = self.descend(units, budget, &mut rng);
        let first = STEP + 1;
        let (mut k, mut found) = (first, 0);
        for round in 0.. {
            if units.is_empty() || round - found > stall {
                break;
            }
            let (mut next, _) = self.shake(units, budget, k.min(units.len()), &mut rng);
            let s = self.descend(&mut next, budget, &mut rng);
            if s < best - 1e-12 {
                (best, found, k) = (s, round, first);
                *units = next;
            } else {
                k = if k >= SHAKE { first } else { k + 1 };
            }
        }
    }

    /// The plan `units` make: everything they leave goes to the tail.
    fn plan(&self, units: &[Vec<Cond>]) -> Plan {
        let state = self.state(units);
        let (tail, added) = self.tail(state.live);
        assert_eq!(
            state.ranks.iter().sum::<u32>() + added,
            required(),
            "the prefix and tail span the published conditions"
        );
        Plan {
            families: units.iter().map(|u| Family::of(u)).collect(),
            tail,
            prefix_ranks: state.ranks,
        }
    }

    /// Everything a prefix at `live` leaves, as few conditions as it can:
    /// each time the one that adds the most rank. Gives back the conditions
    /// and the rank they add.
    fn tail(&self, mut live: [Forest; 32]) -> (Vec<Cond>, u32) {
        let (mut tail, mut added) = (Vec::new(), 0);
        loop {
            let gain = |c: &Cond| {
                let (x, y) = self.bits.ends(c);
                dvs(c.dvs)
                    .filter(|&d| live[d].find(x).0 != live[d].find(y).0)
                    .count()
            };
            let Some((best, gain)) = self
                .cands
                .iter()
                .map(|c| (c, gain(c)))
                .filter(|&(_, g)| g > 0)
                .fold(None, |b: Option<(&Cond, usize)>, (c, g)| {
                    if b.is_none_or(|(_, bg)| g > bg) {
                        Some((c, g))
                    } else {
                        b
                    }
                })
            else {
                return (tail, added);
            };
            let (x, y) = self.bits.ends(best);
            for d in dvs(best.dvs) {
                live[d].union(x, y, best.c);
            }
            tail.push(*best);
            added += gain as u32;
        }
    }
}

/// Where a fill with last blocks `last` stands for DV `d`, on top of `state`.
fn at<'a>(
    last: &'a BTreeMap<usize, (Seen, Alive)>,
    state: &'a State,
    d: usize,
) -> (&'a Seen, &'a Alive) {
    last.get(&d)
        .map_or((&state.last.seen[d], state.last.alive(d)), |l| (&l.0, &l.1))
}

/// `units` without the ones at `out`, in order.
fn without(units: &[Vec<Cond>], out: &[usize]) -> Vec<Vec<Cond>> {
    units
        .iter()
        .enumerate()
        .filter(|(n, _)| !out.contains(n))
        .map(|(_, u)| u.clone())
        .collect()
}

/// The variable neighbourhood search's settings, which reached the best
/// prefix for every target in the fewest shakes. A refill ranks units by the
/// drop per group with probability `PER_GROUP`, by the score otherwise.
const STEP: usize = 6;
const DESCENT: usize = 200;
const SHAKE: usize = 12;
pub const STALL: usize = 30;
const PER_GROUP: f64 = 0.5;

/// What a plan is searched for: lanes per group, groups to spend, and what
/// the target can do for the lanes of one.
#[derive(Clone, Copy)]
pub struct Params {
    pub width: usize,
    pub groups: usize,
    pub ops: Ops,
}

/// The greedy's prefix within `budget`, and whether it reaches every rank.
fn greedy(search: &Search, budget: usize) -> (Vec<Vec<Cond>>, bool) {
    let mut units = Vec::new();
    let state = search.refill(&mut units, budget, true);
    let done = state.ranks.iter().sum::<u32>() == required();
    (units, done)
}

/// Searches for a plan for `p`, until more than `stall` shakes in a row find
/// nothing better: [`STALL`] for the plans, and 0 for a quick look.
pub fn search(p: &Params, stall: usize) -> Plan {
    let search = Search::new(p.width, p.ops);

    let (mut units, done) = greedy(&search, p.groups);
    if !done {
        search.vns(&mut units, p.groups, stall);
    }
    search.plan(&units)
}

/// How much the last block of a message counts against a block of random
/// data in [`objective`].
const PAD_WEIGHT: f64 = 0.02;

/// What a vector search minimizes: how often the tail runs and with how many
/// DVs, on random and on last blocks.
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
