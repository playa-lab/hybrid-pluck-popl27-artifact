use rsdd::builder::bdd::RobddBuilder;
use rsdd::builder::cache::AllIteTable;
use rsdd::builder::BottomUpBuilder;
use rsdd::repr::{BddPtr, DDNNFPtr};

#[derive(Debug, Clone)]
pub struct IntDist {
    pub bits: Vec<BddPtr<'static>>,
}

impl IntDist {
    pub fn new(bits: Vec<BddPtr<'static>>) -> Self {
        IntDist { bits }
    }

    pub fn width(&self) -> usize {
        self.bits.len()
    }

    pub fn bit(&self, i: usize) -> BddPtr<'static> {
        if i < self.bits.len() {
            self.bits[i]
        } else {
            BddPtr::false_ptr()
        }
    }
}

pub fn condition_int_dist(
    x: &IntDist,
    pc: BddPtr<'static>,
    builder: &'static RobddBuilder<'static, AllIteTable<BddPtr<'static>>>,
) -> IntDist {
    if pc.is_true() {
        return x.clone();
    }
    let conditioned_bits: Vec<BddPtr<'static>> =
        x.bits.iter().map(|b| builder.and(*b, pc)).collect();
    IntDist::new(conditioned_bits)
}

pub fn combine_int_dists(
    int_dist_results: &[(IntDist, BddPtr<'static>)],
    builder: &'static RobddBuilder<'static, AllIteTable<BddPtr<'static>>>,
) -> (IntDist, BddPtr<'static>) {
    assert!(!int_dist_results.is_empty());
    let width = int_dist_results[0].0.width();

    let mut result_bits: Vec<BddPtr<'static>> = vec![BddPtr::false_ptr(); width];
    let mut overall_guard = BddPtr::false_ptr();

    for (int_dist, guard) in int_dist_results {
        assert_eq!(width, int_dist.width());
        overall_guard = builder.or(overall_guard, *guard);
        for (result_bit, &dist_bit) in result_bits.iter_mut().zip(int_dist.bits.iter()) {
            let new_bit = builder.and(dist_bit, *guard);
            *result_bit = builder.or(*result_bit, new_bit);
        }
    }

    (IntDist::new(result_bits), overall_guard)
}

pub fn int_dist_eq(
    x: &IntDist,
    y: &IntDist,
    builder: &'static RobddBuilder<'static, AllIteTable<BddPtr<'static>>>,
) -> BddPtr<'static> {
    assert_eq!(x.width(), y.width());

    let mut result = BddPtr::true_ptr();
    for i in 0..x.width() {
        let iff = builder.iff(x.bits[i], y.bits[i]);
        result = builder.and(result, iff);
        if result.is_false() {
            return BddPtr::false_ptr();
        }
    }

    result
}

pub fn int_dist_at_int(
    val: &IntDist,
    i: u64,
    builder: &'static RobddBuilder<'static, AllIteTable<BddPtr<'static>>>,
) -> BddPtr<'static> {
    let mut bdd = BddPtr::true_ptr();
    for bit_idx in 0..val.width() {
        let bit_val = (i >> bit_idx) & 1 == 1;
        let bit_formula = val.bits[bit_idx];
        if bit_val {
            bdd = builder.and(bdd, bit_formula);
        } else {
            let neg = builder.negate(bit_formula);
            bdd = builder.and(bdd, neg);
        }
    }
    bdd
}

pub fn enumerate_int_dist(
    val: &IntDist,
    guard: BddPtr<'static>,
    builder: &'static RobddBuilder<'static, AllIteTable<BddPtr<'static>>>,
) -> Vec<(u64, BddPtr<'static>)> {
    let n = val.width();
    let mut worlds = Vec::new();
    for i in 0..(1u64 << n) {
        let world_bdd = builder.and(int_dist_at_int(val, i, builder), guard);
        if !world_bdd.is_false() {
            worlds.push((i, world_bdd));
        }
    }
    worlds
}

pub fn int_dist_add(
    a: &IntDist,
    b: &IntDist,
    builder: &'static RobddBuilder<'static, AllIteTable<BddPtr<'static>>>,
) -> IntDist {
    assert_eq!(a.width(), b.width());
    let n = a.width();

    let mut result_bits = Vec::with_capacity(n);
    let mut carry = BddPtr::false_ptr();

    for i in 0..n {
        let ai = a.bit(i);
        let bi = b.bit(i);

        let a_xor_b = builder.xor(ai, bi);
        let sum_i = builder.xor(a_xor_b, carry);
        result_bits.push(sum_i);

        let b_or_carry = builder.or(bi, carry);
        let b_and_carry = builder.and(bi, carry);
        carry = builder.ite(ai, b_or_carry, b_and_carry);
    }

    IntDist::new(result_bits)
}

pub fn int_dist_sub(
    a: &IntDist,
    b: &IntDist,
    builder: &'static RobddBuilder<'static, AllIteTable<BddPtr<'static>>>,
) -> IntDist {
    assert_eq!(a.width(), b.width());
    let n = a.width();

    let mut result_bits = Vec::with_capacity(n);
    let mut borrow = BddPtr::false_ptr();

    for i in 0..n {
        let ai = a.bit(i);
        let bi = b.bit(i);

        let a_xor_b = builder.xor(ai, bi);
        let diff_i = builder.xor(a_xor_b, borrow);
        result_bits.push(diff_i);

        let b_and_borrow = builder.and(bi, borrow);
        let b_or_borrow = builder.or(bi, borrow);
        borrow = builder.ite(ai, b_and_borrow, b_or_borrow);
    }

    IntDist::new(result_bits)
}

pub fn int_dist_lt(
    a: &IntDist,
    b: &IntDist,
    builder: &'static RobddBuilder<'static, AllIteTable<BddPtr<'static>>>,
) -> BddPtr<'static> {
    assert_eq!(a.width(), b.width());
    let n = a.width();

    let mut lt = BddPtr::false_ptr();

    for i in 0..n {
        let ai = a.bit(i);
        let bi = b.bit(i);

        let b_and_lt = builder.and(bi, lt);
        let b_or_lt = builder.or(bi, lt);
        lt = builder.ite(ai, b_and_lt, b_or_lt);
    }

    lt
}
