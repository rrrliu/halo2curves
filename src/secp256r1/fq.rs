use core::convert::TryInto;
use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};

use ff::PrimeField;
use rand::RngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

use crate::arithmetic::{adc, mac, sbb};

use pasta_curves::arithmetic::{FieldExt, Group, SqrtRatio};

/// This represents an element of $\mathbb{F}_q$ where
///
/// `q = 0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551`
///
/// is the scalar field of the secp256r1 curve.
// The internal representation of this type is four 64-bit unsigned
// integers in little-endian order. `Fq` values are always in
// Montgomery form; i.e., Fq(a) = aR mod q, with R = 2^256.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fq(pub(crate) [u64; 4]);

/// Constant representing the modulus
/// q = 0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551
const MODULUS: Fq = Fq([
    0xf3b9cac2fc632551,
    0xbce6faada7179e84,
    0xffffffffffffffff,
    0xffffffff00000000,
]);

/// The modulus as u32 limbs.
#[cfg(not(target_pointer_width = "64"))]
const MODULUS_LIMBS_32: [u32; 8] = [
    0xfc63_2551,
    0xf3b9_cac2,
    0xa717_9e84,
    0xbce6_faad,
    0xffff_ffff,
    0xffff_ffff,
    0x0000_0000,
    0xffff_ffff,
];

///Constant representing the modulus as static str
const MODULUS_STR: &str = "0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551";

/// INV = -(q^{-1} mod 2^64) mod 2^64
const INV: u64 = 0xccd1c8aaee00bc4f;


/// R = 2^256 mod q
/// 0xffffffff00000000000000004319055258e8617b0c46353d039cdaaf
const R: Fq = Fq([
    0x0c46353d039cdaaf,
    0x4319055258e8617b,
    0x0000000000000000,
    0xffffffff,
]);

/// R^2 = 2^512 mod q
/// 0x66e12d94f3d956202845b2392b6bec594699799c49bd6fa683244c95be79eea2
const R2: Fq = Fq([
    0x83244c95be79eea2,
    0x4699799c49bd6fa6,
    0x2845b2392b6bec59,
    0x66e12d94f3d95620,
]);

/// R^3 = 2^768 mod q
/// 0x503a54e76407be652543b9246ba5e93f111f28ae0c0555c9ac8ebec90b65a624
const R3: Fq = Fq([
    0xac8ebec90b65a624,
    0x111f28ae0c0555c9,
    0x2543b9246ba5e93f,
    0x503a54e76407be65,
]);

/// `GENERATOR = 7 mod r` is a generator of the `q - 1` order multiplicative
/// subgroup, or in other words a primitive root of the field.
const GENERATOR: Fq = Fq::from_raw([0x07, 0x00, 0x00, 0x00]);

/// GENERATOR^t where t * 2^s + 1 = r
/// with t odd. In other words, this
/// is a 2^s root of unity.
/// `0xffc97f062a770992ba807ace842a3dfc1546cad004378daf0592d7fbb41e6602`
const ROOT_OF_UNITY: Fq = Fq::from_raw([
    0x0592d7fbb41e6602,
    0x1546cad004378daf,
    0xba807ace842a3dfc,
    0xffc97f062a770992,
]);

/// 1 / ROOT_OF_UNITY mod q
const ROOT_OF_UNITY_INV: Fq = Fq::from_raw([
    0x379c7f0657c73764,
    0xe3ac117c794c4137,
    0xc645fa0458131cae,
    0xa0a66a5562d46f2a,
]);

/// 1 / 2 mod q
const TWO_INV: Fq = Fq::from_raw([
    0x79dce5617e3192a9,
    0xde737d56d38bcf42,
    0x7fffffffffffffff,
    0x7fffffff80000000,
]);

const ZETA: Fq = Fq::zero();
const DELTA: Fq = Fq::zero();

use crate::{
    field_arithmetic, field_common, field_specific, impl_add_binop_specify_output,
    impl_binops_additive, impl_binops_additive_specify_output, impl_binops_multiplicative,
    impl_binops_multiplicative_mixed, impl_sub_binop_specify_output,
};
impl_binops_additive!(Fq, Fq);
impl_binops_multiplicative!(Fq, Fq);
field_common!(
    Fq,
    MODULUS,
    INV,
    MODULUS_STR,
    TWO_INV,
    ROOT_OF_UNITY_INV,
    DELTA,
    ZETA,
    R,
    R2,
    R3
);
field_arithmetic!(Fq, MODULUS, INV, dense);

impl Fq {
    pub const fn size() -> usize {
        32
    }
}

impl ff::Field for Fq {
    fn random(mut rng: impl RngCore) -> Self {
        Self::from_u512([
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ])
    }

    fn zero() -> Self {
        Self::zero()
    }

    fn one() -> Self {
        Self::one()
    }

    fn double(&self) -> Self {
        self.double()
    }

    #[inline(always)]
    fn square(&self) -> Self {
        self.square()
    }

    /// Computes the square root of this element, if it exists.
    fn sqrt(&self) -> CtOption<Self> {
        crate::arithmetic::sqrt_tonelli_shanks(self, &<Self as SqrtRatio>::T_MINUS1_OVER2)
    }

    /// Computes the multiplicative inverse of this element,
    /// failing if the element is zero.
    fn invert(&self) -> CtOption<Self> {
        let tmp = self.pow_vartime(&[
            0xf3b9cac2fc63254f,
            0xbce6faada7179e84,
            0xffffffffffffffff,
            0xffffffff00000000,
        ]);

        CtOption::new(tmp, !self.ct_eq(&Self::zero()))
    }

    fn pow_vartime<S: AsRef<[u64]>>(&self, exp: S) -> Self {
        let mut res = Self::one();
        let mut found_one = false;
        for e in exp.as_ref().iter().rev() {
            for i in (0..64).rev() {
                if found_one {
                    res = res.square();
                }

                if ((*e >> i) & 1) == 1 {
                    found_one = true;
                    res *= self;
                }
            }
        }
        res
    }
}

impl ff::PrimeField for Fq {
    type Repr = [u8; 32];

    const NUM_BITS: u32 = 256;
    const CAPACITY: u32 = 255;
    const S: u32 = 4;

    fn from_repr(repr: Self::Repr) -> CtOption<Self> {
        let mut tmp = Fq([0, 0, 0, 0]);

        tmp.0[0] = u64::from_le_bytes(repr[0..8].try_into().unwrap());
        tmp.0[1] = u64::from_le_bytes(repr[8..16].try_into().unwrap());
        tmp.0[2] = u64::from_le_bytes(repr[16..24].try_into().unwrap());
        tmp.0[3] = u64::from_le_bytes(repr[24..32].try_into().unwrap());

        // Try to subtract the modulus
        let (_, borrow) = sbb(tmp.0[0], MODULUS.0[0], 0);
        let (_, borrow) = sbb(tmp.0[1], MODULUS.0[1], borrow);
        let (_, borrow) = sbb(tmp.0[2], MODULUS.0[2], borrow);
        let (_, borrow) = sbb(tmp.0[3], MODULUS.0[3], borrow);

        // If the element is smaller than MODULUS then the
        // subtraction will underflow, producing a borrow value
        // of 0xffff...ffff. Otherwise, it'll be zero.
        let is_some = (borrow as u8) & 1;

        // Convert to Montgomery form by computing
        // (a.R^0 * R^2) / R = a.R
        tmp *= &R2;

        CtOption::new(tmp, Choice::from(is_some))
    }

    fn to_repr(&self) -> Self::Repr {
        // Turn into canonical form by computing
        // (a.R) / R = a
        let tmp = Fq::montgomery_reduce(&[self.0[0], self.0[1], self.0[2], self.0[3], 0, 0, 0, 0]);

        let mut res = [0; 32];
        res[0..8].copy_from_slice(&tmp.0[0].to_le_bytes());
        res[8..16].copy_from_slice(&tmp.0[1].to_le_bytes());
        res[16..24].copy_from_slice(&tmp.0[2].to_le_bytes());
        res[24..32].copy_from_slice(&tmp.0[3].to_le_bytes());

        res
    }

    fn is_odd(&self) -> Choice {
        Choice::from(self.to_repr()[0] & 1)
    }

    fn multiplicative_generator() -> Self {
        GENERATOR
    }

    fn root_of_unity() -> Self {
        ROOT_OF_UNITY
    }
}

impl SqrtRatio for Fq {
    const T_MINUS1_OVER2: [u64; 4] = [
        0x279dce5617e3192a,
        0xfde737d56d38bcf4,
        0x07ffffffffffffff,
        0x07fffffff8000000,
    ];

    fn get_lower_32(&self) -> u32 {
        let tmp = Fq::montgomery_reduce(&[self.0[0], self.0[1], self.0[2], self.0[3], 0, 0, 0, 0]);
        tmp.0[0] as u32
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ff::Field;
    use rand_core::OsRng;

    #[test]
    fn test_sqrt() {
        // NB: TWO_INV is standing in as a "random" field element
        let v = (Fq::TWO_INV).square().sqrt().unwrap();
        assert!(v == Fq::TWO_INV || (-v) == Fq::TWO_INV);

        for _ in 0..10000 {
            let a = Fq::random(OsRng);
            let mut b = a;
            b = b.square();

            let b = b.sqrt().unwrap();
            let mut negb = b;
            negb = negb.neg();

            assert!(a == b || a == negb);
        }
    }

    #[test]
    fn test_root_of_unity() {
        assert_eq!(
            Fq::root_of_unity().pow_vartime(&[1 << Fq::S, 0, 0, 0]),
            Fq::one()
        );
    }

    #[test]
    fn test_inv_root_of_unity() {
        assert_eq!(Fq::ROOT_OF_UNITY_INV, Fq::root_of_unity().invert().unwrap());
    }

    #[test]
    fn test_field() {
        crate::tests::field::random_field_tests::<Fq>("secp256r1 scalar".to_string());
    }

    #[test]
    fn test_serialization() {
        crate::tests::field::random_serialization_test::<Fq>("secp256r1 scalar".to_string());
    }
}
