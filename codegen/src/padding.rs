//! The last block of a message, which a vector prefix is scored on too.
//!
//! A message's last block is its leftover data, a 1 bit, zeros and the
//! length. Much of it is fixed, so a condition there may fail always or
//! never, and a prefix good on random data can leave many DVs alive.
//!
//! The chance a DV survives is worked out exactly, per length mod 64. Its
//! conditions are sums of its published ones, linear over the free bits. A
//! sum the free bits do not reach is fixed: the DV survives only if all such
//! sums come out right, and then with `2^-r` for the rank `r` of the rest.
//! A DV has at most 15 published conditions, so a sum fits in a `u16`.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::solve::{Cond, Plan, dvs};
use crate::ubc::UBCS;

/// How many residues of a message's length mod 64 there are, each ending
/// the message in a block of its own shape.
const RESIDUES: usize = 64;

/// The length's bits from this one up are taken to be zero: messages are
/// under 8 KiB. The bits below it and above the residue's are free.
const LENGTH_BITS: u32 = 16;

/// `2^-n`, exactly.
pub fn half(n: u32) -> f64 {
    f64::from_bits((1023 - u64::from(n)) << 52)
}

/// A linear form over the 512 bits of a block.
type Form = [u64; 8];

fn xor(a: &Form, b: &Form) -> Form {
    std::array::from_fn(|k| a[k] ^ b[k])
}

/// The highest set bit of a form.
fn pivot(f: &Form) -> usize {
    let k = (0..8).rev().find(|&k| f[k] != 0).expect("a nonzero form");
    k * 64 + 63 - f[k].leading_zeros() as usize
}

/// Bit `b` of `w[t]`, as a form over the block: the first 16 words are the
/// block itself, and the expansion is linear.
fn forms() -> Vec<[Form; 32]> {
    let mut w: Vec<[Form; 32]> = (0..16)
        .map(|t| {
            std::array::from_fn(|b| {
                let bit = t * 32 + b;
                let mut f = [0; 8];
                f[bit / 64] |= 1 << (bit % 64);
                f
            })
        })
        .collect();
    for t in 16..80 {
        let word = std::array::from_fn(|b| {
            // A rotation left by one takes bit `b` from bit `b - 1`.
            let s = (b + 31) % 32;
            xor(
                &xor(&w[t - 3][s], &w[t - 8][s]),
                &xor(&w[t - 14][s], &w[t - 16][s]),
            )
        });
        w.push(word);
    }
    w
}

/// What the last block of a message of length `r` mod 64 fixes: which of
/// its bits are free, and the value of the rest.
fn shape(r: usize) -> (Form, Form) {
    let (mut free, mut value) = ([0u64; 8], [0u64; 8]);
    let set = |form: &mut Form, word: usize, bit: u32| {
        let k = word * 32 + bit as usize;
        form[k / 64] |= 1 << (k % 64);
    };
    // Byte `k` is loaded big-endian: bits 31 to 24 of word `k / 4` when
    // `k % 4` is 0, and so on down.
    let byte_bit = |k: usize, q: u32| (k / 4, 8 * (3 - (k % 4) as u32) + q);
    if r < 56 {
        for k in 0..r {
            for q in 0..8 {
                let (word, bit) = byte_bit(k, q);
                set(&mut free, word, bit);
            }
        }
        let (word, bit) = byte_bit(r, 7);
        set(&mut value, word, bit);
    }
    // The length in bits ends the block: its low three bits are zero, the
    // next six the residue, and the next few free.
    for bit in 3..9 {
        if (r >> (bit - 3)) & 1 == 1 {
            set(&mut value, 15, bit);
        }
    }
    for bit in 9..LENGTH_BITS {
        set(&mut free, 15, bit);
    }
    (free, value)
}

/// A subspace of one DV's condition space in reduced echelon form: the row
/// whose highest bit is `p` is `rows[p]`, and no other row has bit `p`.
#[derive(Clone, PartialEq, Eq)]
struct Sub {
    rows: [u16; 16],
    /// The positions that have a row.
    pivots: u16,
    rank: u32,
}

impl Sub {
    fn new() -> Self {
        Self {
            rows: [0; 16],
            pivots: 0,
            rank: 0,
        }
    }

    /// Whether `v` is outside the subspace, so that adding it raises the rank.
    fn grows(&self, v: u16) -> bool {
        self.reduce(v) != 0
    }

    /// `v` with every pivot cleared, which is the same for all of `v` modulo
    /// the subspace. No row has another's pivot, so each pivot `v` has is
    /// cleared by its own row alone, in any order.
    fn reduce(&self, mut v: u16) -> u16 {
        let mut at = v & self.pivots;
        while at != 0 {
            v ^= self.rows[at.trailing_zeros() as usize];
            at &= at - 1;
        }
        v
    }

    /// Adds `v`, and whether it was new.
    fn insert(&mut self, v: u16) -> bool {
        let v = self.reduce(v);
        if v == 0 {
            return false;
        }
        let p = 15 - v.leading_zeros() as usize;
        // Keep the form reduced: clear the new pivot from the other rows.
        for row in &mut self.rows {
            if *row >> p & 1 == 1 {
                *row ^= v;
            }
        }
        self.rows[p] = v;
        self.pivots |= 1 << p;
        self.rank += 1;
        true
    }
}

/// What a residue does to one DV: `fixed` holds the sums of its conditions
/// that the free bits do not reach, and `pass` those of them that come out
/// as the published conditions require.
#[derive(PartialEq, Eq)]
struct Class {
    fixed: Sub,
    pass: Sub,
}

/// One DV's published conditions, and what each residue does to them.
struct Dv {
    /// For each published condition, its two bits.
    edges: Vec<((usize, u32), (usize, u32))>,
    /// The class of each residue, or `None` where the free bits reach every
    /// sum.
    class_of: [Option<usize>; RESIDUES],
    classes: Vec<Class>,
}

impl Dv {
    fn new(d: usize, forms: &[[Form; 32]], shapes: &[(Form, Form)]) -> Self {
        let published: Vec<_> = UBCS.iter().filter(|u| u.dvs >> d & 1 == 1).collect();
        assert!(
            published.len() <= 16,
            "DV {d} has more conditions than a u16 holds"
        );
        let mut classes: Vec<Class> = Vec::new();
        let class_of = std::array::from_fn(|r| {
            let (free, value) = &shapes[r];
            // Eliminate the free part of each published condition, tracking
            // which of them each row sums. A row with nothing free left is a
            // fixed sum.
            let mut basis: Vec<(Form, u16)> = Vec::new();
            let mut fixed = Sub::new();
            let mut wrong = 0u16;
            for (e, u) in published.iter().enumerate() {
                let form = xor(&forms[u.i][u.a as usize], &forms[u.j][u.b as usize]);
                let mut row: Form = std::array::from_fn(|k| form[k] & free[k]);
                let mut tag = 1u16 << e;
                let parity: u32 = (0..8)
                    .map(|k| (form[k] & !free[k] & value[k]).count_ones())
                    .sum();
                // What the fixed bits alone make the XOR, against what it
                // must be.
                if parity & 1 != u.c {
                    wrong |= 1 << e;
                }
                for (b, t) in &basis {
                    let p = pivot(b);
                    if row[p / 64] >> (p % 64) & 1 == 1 {
                        row = xor(&row, b);
                        tag ^= t;
                    }
                }
                if row == [0; 8] {
                    fixed.insert(tag);
                } else {
                    basis.push((row, tag));
                }
            }
            if fixed.rank == 0 {
                return None;
            }
            // A fixed sum comes out right when it holds an even number of
            // conditions the fixed bits get wrong.
            let rows: Vec<u16> = fixed.rows.iter().copied().filter(|&v| v != 0).collect();
            let right = |v: &u16| (v & wrong).count_ones().is_multiple_of(2);
            let mut pass = Sub::new();
            for &v in rows.iter().filter(|v| right(v)) {
                pass.insert(v);
            }
            let odd: Vec<u16> = rows.iter().copied().filter(|v| !right(v)).collect();
            for &v in odd.iter().skip(1) {
                pass.insert(v ^ odd[0]);
            }
            let class = Class { fixed, pass };
            Some(classes.iter().position(|c| *c == class).unwrap_or_else(|| {
                classes.push(class);
                classes.len() - 1
            }))
        });
        Self {
            edges: published.iter().map(|u| ((u.i, u.a), (u.j, u.b))).collect(),
            class_of,
            classes,
        }
    }

    /// A condition of this DV as a sum of its published ones: the path
    /// between its two bits in the forest they make.
    fn sum(&self, c: &Cond) -> u16 {
        let (from, to) = ((c.i, c.a), (c.j, c.b));
        let mut came: HashMap<(usize, u32), ((usize, u32), usize)> = HashMap::new();
        let mut seen = HashSet::from([from]);
        let mut queue = VecDeque::from([from]);
        while let Some(v) = queue.pop_front() {
            for (e, &(x, y)) in self.edges.iter().enumerate() {
                let next = match v {
                    _ if x == v => y,
                    _ if y == v => x,
                    _ => continue,
                };
                if seen.insert(next) {
                    came.insert(next, (v, e));
                    queue.push_back(next);
                }
            }
        }
        let mut sum = 0u16;
        let mut v = to;
        while v != from {
            let (prev, e) = came[&v];
            sum ^= 1 << e;
            v = prev;
        }
        sum
    }
}

/// What the prefix has reached of one DV's sums, modulo each class's fixed
/// and passing ones.
#[derive(Clone)]
pub struct Seen {
    fixed: Vec<Sub>,
    pass: Vec<Sub>,
}

/// Each DV's chance to survive on each residue.
pub type Alive = [f64; RESIDUES];

/// Where the last blocks stand after a prefix, per DV and per residue, with
/// totals so that a trial costs only the DVs it changes.
pub struct Last {
    pub seen: Vec<Seen>,
    alive: [Alive; 32],
    totals: Totals,
}

/// Per residue: the product of `1 - alive` over the DVs where it is not
/// zero, how many DVs make it zero, and the sum of `alive`.
#[derive(Clone)]
pub struct Totals {
    none: [f64; RESIDUES],
    certain: [u32; RESIDUES],
    sum: [f64; RESIDUES],
}

impl Totals {
    fn of(alive: &[Alive; 32]) -> Self {
        let mut this = Totals {
            none: [1.0; RESIDUES],
            certain: [0; RESIDUES],
            sum: [0.0; RESIDUES],
        };
        for r in 0..RESIDUES {
            let (mut none, mut certain, mut sum) = (1.0, 0, 0.0);
            for row in alive {
                if row[r] == 1.0 {
                    certain += 1;
                } else {
                    none *= 1.0 - row[r];
                }
                sum += row[r];
            }
            (this.none[r], this.certain[r], this.sum[r]) = (none, certain, sum);
        }
        this
    }

    /// Replaces one residue's chance `old` of some DV by `new`.
    fn step((none, certain, sum): (&mut f64, &mut u32, &mut f64), old: f64, new: f64) {
        if old == 1.0 {
            *certain -= 1;
        } else {
            *none /= 1.0 - old;
        }
        if new == 1.0 {
            *certain += 1;
        } else {
            *none *= 1.0 - new;
        }
        *sum += new - old;
    }

    /// Replaces a DV's chances `old` by `new`.
    pub fn swap(&mut self, old: &Alive, new: &Alive) {
        for r in 0..RESIDUES {
            let at = (&mut self.none[r], &mut self.certain[r], &mut self.sum[r]);
            Self::step(at, old[r], new[r]);
        }
    }

    /// [`Last::tail`], with each DV's chances in `changed` swapped from the
    /// first of its pair to the second.
    pub fn tail<'a>(
        &self,
        changed: impl Iterator<Item = (&'a Alive, &'a Alive)> + Clone,
    ) -> (f64, f64) {
        let (mut enters, mut dvs) = (0.0, 0.0);
        for r in 0..RESIDUES {
            let (mut none, mut certain, mut sum) = (self.none[r], self.certain[r], self.sum[r]);
            for (old, new) in changed.clone() {
                Self::step((&mut none, &mut certain, &mut sum), old[r], new[r]);
            }
            enters += if certain > 0 { 1.0 } else { 1.0 - none };
            dvs += sum;
        }
        (enters / RESIDUES as f64, dvs / RESIDUES as f64)
    }
}

impl Last {
    /// Replaces DV `d`. The residues' totals are stale until [`Last::total`].
    pub fn put(&mut self, d: usize, seen: Seen, alive: Alive) {
        self.seen[d] = seen;
        self.alive[d] = alive;
    }

    /// Works the residues' totals out again from every DV.
    pub fn total(&mut self) {
        self.totals = Totals::of(&self.alive);
    }

    /// The residues' totals.
    pub fn totals(&self) -> &Totals {
        &self.totals
    }

    /// DV `d`'s chance to survive on each residue.
    pub fn alive(&self, d: usize) -> &Alive {
        &self.alive[d]
    }

    /// Each DV's chance to survive a last block, over the residues.
    pub fn means(&self) -> [f64; 32] {
        std::array::from_fn(|d| mean(&self.alive[d]))
    }

    /// How often a last block enters the tail, taking the DVs to survive
    /// independently as for random data, and how many DVs it carries in.
    pub fn tail(&self) -> (f64, f64) {
        self.totals.tail(std::iter::empty())
    }
}

/// A DV's chance to survive a last block, over the residues.
pub fn mean(alive: &Alive) -> f64 {
    alive.iter().sum::<f64>() / RESIDUES as f64
}

/// The published conditions of every DV, and what each residue does to them.
pub struct Exact {
    dvs: Vec<Dv>,
    /// Each candidate's sums, which the search asks for often.
    sums: HashMap<(Cond, usize), u16>,
}

impl Exact {
    pub fn new(cands: &[Cond]) -> Self {
        let forms = forms();
        let shapes: Vec<(Form, Form)> = (0..RESIDUES).map(shape).collect();
        let all: Vec<Dv> = (0..32).map(|d| Dv::new(d, &forms, &shapes)).collect();
        let sums = cands
            .iter()
            .flat_map(|c| {
                dvs(c.dvs)
                    .map(|d| ((*c, d), all[d].sum(c)))
                    .collect::<Vec<_>>()
            })
            .collect();
        Self { dvs: all, sums }
    }

    fn sum(&self, c: &Cond, d: usize) -> u16 {
        self.sums
            .get(&(*c, d))
            .copied()
            .unwrap_or_else(|| self.dvs[d].sum(c))
    }

    /// The last blocks before anything is checked.
    pub fn empty(&self) -> Last {
        Last {
            seen: self
                .dvs
                .iter()
                .map(|dv| Seen {
                    fixed: vec![Sub::new(); dv.classes.len()],
                    pass: vec![Sub::new(); dv.classes.len()],
                })
                .collect(),
            alive: [[1.0; RESIDUES]; 32],
            totals: Totals {
                none: [1.0; RESIDUES],
                certain: [32; RESIDUES],
                sum: [32.0; RESIDUES],
            },
        }
    }

    /// DV `d` once `c` is checked too, from where `seen` has it.
    pub fn seen_after(&self, seen: &Seen, d: usize, c: &Cond) -> Seen {
        let v = self.sum(c, d);
        let mut seen = seen.clone();
        for (k, class) in self.dvs[d].classes.iter().enumerate() {
            seen.fixed[k].insert(class.fixed.reduce(v));
            seen.pass[k].insert(class.pass.reduce(v));
        }
        seen
    }

    /// [`Exact::alive`] after [`Exact::seen_after`], without a copy.
    pub fn alive_after(&self, seen: &Seen, d: usize, c: &Cond, rank: u32) -> Alive {
        let v = self.sum(c, d);
        // A class for each residue at most, so the ranks fit on the stack.
        let mut ranks = [(0, 0); RESIDUES];
        for (k, class) in self.dvs[d].classes.iter().enumerate() {
            let after = |s: &Sub, class: &Sub| s.rank + u32::from(s.grows(class.reduce(v)));
            ranks[k] = (
                after(&seen.fixed[k], &class.fixed),
                after(&seen.pass[k], &class.pass),
            );
        }
        self.chances(d, rank, |k| ranks[k])
    }

    /// Checks `c` for DV `d` in `last` as well, where the prefix then reaches
    /// rank `rank` for it. The residues' totals are left to the caller.
    pub fn add(&self, last: &mut Last, d: usize, c: &Cond, rank: u32) {
        let seen = self.seen_after(&last.seen[d], d, c);
        let alive = self.alive(&seen, d, rank);
        last.put(d, seen, alive);
    }

    /// DV `d`'s chance to survive on each residue, from where `seen` has it
    /// and where the prefix reaches rank `rank` for it.
    pub fn alive(&self, seen: &Seen, d: usize, rank: u32) -> Alive {
        self.chances(d, rank, |k| (seen.fixed[k].rank, seen.pass[k].rank))
    }

    fn chances(&self, d: usize, rank: u32, ranks: impl Fn(usize) -> (u32, u32)) -> Alive {
        std::array::from_fn(|r| match self.dvs[d].class_of[r] {
            None => half(rank),
            Some(k) => {
                // Equal ranks modulo the fixed sums and modulo the passing
                // ones mean that no fixed sum the prefix reaches fails; the
                // rest is up to the free bits.
                let (fixed, pass) = ranks(k);
                if fixed == pass { half(fixed) } else { 0.0 }
            }
        })
    }
}

/// Where the last blocks stand after checking `conds`.
fn last_of(conds: impl Iterator<Item = Cond>) -> Last {
    let exact = Exact::new(&[]);
    let mut last = exact.empty();
    let mut spans = vec![Sub::new(); 32];
    for c in conds {
        for d in dvs(c.dvs) {
            spans[d].insert(exact.sum(&c, d));
            exact.add(&mut last, d, &c, spans[d].rank);
        }
    }
    last.total();
    last
}

/// How often the last blocks enter the tail after `plan`'s prefix, and how
/// many DVs they carry in, for the report.
pub fn tail_of(plan: &Plan) -> (f64, f64) {
    last_of(plan.families.iter().flat_map(|f| f.conds())).tail()
}

/// The message lengths, mod 64, whose every last block `plan`'s prefix
/// sends into the tail.
pub fn cliffs(plan: &Plan) -> Vec<usize> {
    let last = last_of(plan.families.iter().flat_map(|f| f.conds()));
    (0..RESIDUES)
        .filter(|&r| last.totals.certain[r] > 0)
        .collect()
}

#[cfg(test)]
mod tests {
    use rand::rngs::Xoshiro256PlusPlus;
    use rand::{RngExt as _, SeedableRng as _};

    use super::*;
    use crate::solve::{Ops, Params, search};

    /// The last block of a message of length `r` mod 64, drawn at random:
    /// its data bytes, and its length under 8 KiB.
    fn block(rng: &mut Xoshiro256PlusPlus, r: usize) -> [u32; 80] {
        let mut bytes = [0u8; 64];
        if r < 56 {
            rng.fill(&mut bytes[..r]);
            bytes[r] = 0x80;
        }
        let len = 64 * rng.random_range(0..128u64) + r as u64;
        bytes[56..].copy_from_slice(&(len * 8).to_be_bytes());
        let mut w = [0u32; 80];
        for (t, word) in bytes.as_chunks::<4>().0.iter().enumerate() {
            w[t] = u32::from_be_bytes(*word);
        }
        for t in 16..80 {
            w[t] = (w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16]).rotate_left(1);
        }
        w
    }

    /// The worked-out chances against counting, for a real prefix: those
    /// that come out as 0 or 1 must hold for every block, and the rest must
    /// be close. The draw is seeded, so this cannot flake.
    #[test]
    fn exact_chances_match_counting() {
        const DRAWS: usize = 4000;
        let ops = Ops { lane_bit: true };
        let conds: Vec<Cond> = search(
            &Params {
                width: 4,
                groups: 20,
                ops,
            },
            0,
        )
        .families
        .iter()
        .flat_map(|f| f.conds())
        .collect();
        let last = last_of(conds.iter().copied());
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x0123_4567_89ab_cdef);
        for r in 0..RESIDUES {
            let mut survived = [0usize; 32];
            for _ in 0..DRAWS {
                let w = block(&mut rng, r);
                let mut alive = !0u32;
                for c in &conds {
                    if (w[c.i] >> c.a ^ w[c.j] >> c.b) & 1 != c.c {
                        alive &= !c.dvs;
                    }
                }
                for (d, n) in survived.iter_mut().enumerate() {
                    *n += (alive >> d & 1) as usize;
                }
            }
            for (d, &n) in survived.iter().enumerate() {
                let (want, got) = (last.alive[d][r], n as f64 / DRAWS as f64);
                if want == 0.0 || want == 1.0 {
                    assert_eq!(got, want, "DV {d}, residue {r}");
                } else {
                    assert!(
                        (got - want).abs() < 0.05,
                        "DV {d}, residue {r}: {got} for {want}"
                    );
                }
            }
        }
    }
}
