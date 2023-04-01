#[macro_export]
macro_rules! batch_add {
    () => {
        fn batch_add<const COMPLETE: bool, const LOAD_POINTS: bool>(
            points: &mut [Self],
            output_indices: &[u32],
            num_points: usize,
            offset: usize,
            bases: &[Self],
            base_positions: &[u32],
        ) {
            // assert!(Self::constant_a().is_zero());

            let get_point = |point_data: u32| -> Self {
                let negate = point_data & 0x80000000 != 0;
                let base_idx = (point_data & 0x7FFFFFFF) as usize;
                if negate {
                    bases[base_idx].neg()
                } else {
                    bases[base_idx]
                }
            };

            // Affine addition formula (P != Q):
            // - lambda = (y_2 - y_1) / (x_2 - x_1)
            // - x_3 = lambda^2 - (x_2 + x_1)
            // - y_3 = lambda * (x_1 - x_3) - y_1

            // Batch invert accumulator
            let mut acc = Self::Base::one();

            for i in (0..num_points).step_by(2) {
                // Where that result of the point addition will be stored
                let out_idx = output_indices[i >> 1] as usize - offset;

                #[cfg(all(feature = "prefetch", target_arch = "x86_64"))]
                if i < num_points - 2 {
                    if LOAD_POINTS {
                        $crate::prefetch::<Self>(bases, base_positions[i + 2] as usize);
                        $crate::prefetch::<Self>(bases, base_positions[i + 3] as usize);
                    }
                    $crate::prefetch::<Self>(
                        points,
                        output_indices[(i >> 1) + 1] as usize - offset,
                    );
                }
                if LOAD_POINTS {
                    points[i] = get_point(base_positions[i]);
                    points[i + 1] = get_point(base_positions[i + 1]);
                }

                if COMPLETE {
                    // Nothing to do here if one of the points is zero
                    if (points[i].is_identity() | points[i + 1].is_identity()).into() {
                        continue;
                    }

                    if points[i].x == points[i + 1].x {
                        if points[i].y == points[i + 1].y {
                            // Point doubling (P == Q)
                            // - s = (3 * x^2) / (2 * y)
                            // - x_2 = s^2 - (2 * x)
                            // - y_2 = s * (x - x_2) - y

                            // (2 * x)
                            points[out_idx].x = points[i].x + points[i].x;
                            // x^2
                            let xx = points[i].x.square();
                            // (2 * y)
                            points[i + 1].x = points[i].y + points[i].y;
                            // (3 * x^2) * acc
                            points[i + 1].y = (xx + xx + xx) * acc;
                            // acc * (2 * y)
                            acc *= points[i + 1].x;
                            continue;
                        } else {
                            // Zero
                            points[i] = Self::identity();
                            points[i + 1] = Self::identity();
                            continue;
                        }
                    }
                }

                // (x_2 + x_1)
                points[out_idx].x = points[i].x + points[i + 1].x;
                // (x_2 - x_1)
                points[i + 1].x -= points[i].x;
                // (y2 - y1) * acc
                points[i + 1].y = (points[i + 1].y - points[i].y) * acc;
                // acc * (x_2 - x_1)
                acc *= points[i + 1].x;
            }

            // Batch invert
            if COMPLETE {
                if (!acc.is_zero()).into() {
                    acc = acc.invert().unwrap();
                }
            } else {
                acc = acc.invert().unwrap();
            }

            for i in (0..num_points).step_by(2).rev() {
                // Where that result of the point addition will be stored
                let out_idx = output_indices[i >> 1] as usize - offset;

                #[cfg(all(feature = "prefetch", target_arch = "x86_64"))]
                if i > 0 {
                    $crate::prefetch::<Self>(
                        points,
                        output_indices[(i >> 1) - 1] as usize - offset,
                    );
                }

                if COMPLETE {
                    // points[i] is zero so the sum is points[i + 1]
                    if points[i].is_identity().into() {
                        points[out_idx] = points[i + 1];
                        continue;
                    }
                    // points[i + 1] is zero so the sum is points[i]
                    if points[i + 1].is_identity().into() {
                        points[out_idx] = points[i];
                        continue;
                    }
                }

                // lambda
                points[i + 1].y *= acc;
                // acc * (x_2 - x_1)
                acc *= points[i + 1].x;
                // x_3 = lambda^2 - (x_2 + x_1)
                points[out_idx].x = points[i + 1].y.square() - points[out_idx].x;
                // y_3 = lambda * (x_1 - x_3) - y_1
                points[out_idx].y =
                    points[i + 1].y * (points[i].x - points[out_idx].x) - points[i].y;
            }
        }
    };
}

#[macro_export]
macro_rules! new_curve_impl {
    (($($privacy:tt)*),
    $name:ident,
    $name_affine:ident,
    $name_compressed:ident,
    $compressed_size:expr,
    $base:ident,
    $scalar:ident,
    $generator:expr,
    $constant_a:expr,
    $constant_b:expr,
    $curve_id:literal,
    ) => {

        #[derive(Copy, Clone, Debug)]
        $($privacy)* struct $name {
            pub x: $base,
            pub y: $base,
            pub z: $base,
        }

        #[derive(Copy, Clone, PartialEq)]
        $($privacy)* struct $name_affine {
            pub x: $base,
            pub y: $base,
        }

        #[derive(Copy, Clone)]
        $($privacy)* struct $name_compressed([u8; $compressed_size]);


        impl $name {
            pub fn generator() -> Self {
                let generator = $name_affine::generator();
                Self {
                    x: generator.x,
                    y: generator.y,
                    z: $base::one(),
                }
            }

            const fn curve_constant_a() -> $base {
                $name_affine::curve_constant_a()
            }

            const fn curve_constant_b() -> $base {
                $name_affine::curve_constant_b()
            }

            #[inline]
            fn curve_constant_3b() -> $base {
                lazy_static::lazy_static! {
                        static ref CONST_3B: $base = $constant_b + $constant_b + $constant_b;
                }
                *CONST_3B
            }

            fn mul_by_3b(input: &$base) -> $base {
                if $name::CURVE_ID == "bn256"{
                    input.double().double().double() + input
                } else {
                    input * $name::curve_constant_3b()
                }
            }
        }

        impl $name_affine {
            pub fn generator() -> Self {
                Self {
                    x: $generator.0,
                    y: $generator.1,
                }
            }

            const fn curve_constant_a() -> $base {
                $constant_a
            }

            const fn curve_constant_b() -> $base {
                $constant_b
            }

            pub fn random(mut rng: impl RngCore) -> Self {
                loop {
                    let x = $base::random(&mut rng);
                    let ysign = (rng.next_u32() % 2) as u8;

                    let x3 = x.square() * x;
                    let y = (x3 + $name::curve_constant_a() * x + $name::curve_constant_b()).sqrt();
                    if let Some(y) = Option::<$base>::from(y) {
                        let sign = y.to_bytes()[0] & 1;
                        let y = if ysign ^ sign == 0 { y } else { -y };

                        let p = $name_affine {
                            x,
                            y,
                        };


                        use $crate::group::cofactor::CofactorGroup;
                        let p = p.to_curve();
                        return p.clear_cofactor().to_affine()
                    }
                }
            }
        }

        // Compressed

        impl std::fmt::Debug for $name_compressed {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                self.0[..].fmt(f)
            }
        }

        impl Default for $name_compressed {
            fn default() -> Self {
                $name_compressed([0; $compressed_size])
            }
        }

        impl AsRef<[u8]> for $name_compressed {
            fn as_ref(&self) -> &[u8] {
                &self.0
            }
        }

        impl AsMut<[u8]> for $name_compressed {
            fn as_mut(&mut self) -> &mut [u8] {
                &mut self.0
            }
        }


        // Jacobian implementations

        impl<'a> From<&'a $name_affine> for $name {
            fn from(p: &'a $name_affine) -> $name {
                p.to_curve()
            }
        }

        impl From<$name_affine> for $name {
            fn from(p: $name_affine) -> $name {
                p.to_curve()
            }
        }

        impl Default for $name {
            fn default() -> $name {
                $name::identity()
            }
        }

        impl subtle::ConstantTimeEq for $name {
            fn ct_eq(&self, other: &Self) -> Choice {
                // Is (x, y, z) equal to (x', y, z') when converted to affine?	
                // => (x/z , y/z) equal to (x'/z' , y'/z')	
                // => (xz' == x'z) & (yz' == y'z)	
                let x1 = self.x * other.z;	
                let y1 = self.y * other.z;	
                let x2 = other.x * self.z;	
                let y2 = other.y * self.z;

                let self_is_zero = self.is_identity();
                let other_is_zero = other.is_identity();

                (self_is_zero & other_is_zero) // Both point at infinity
                            | ((!self_is_zero) & (!other_is_zero) & x1.ct_eq(&x2) & y1.ct_eq(&y2))
                // Neither point at infinity, coordinates are the same
            }

        }

        impl subtle::ConditionallySelectable for $name {
            fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
                $name {
                    x: $base::conditional_select(&a.x, &b.x, choice),
                    y: $base::conditional_select(&a.y, &b.y, choice),
                    z: $base::conditional_select(&a.z, &b.z, choice),
                }
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.ct_eq(other).into()
            }
        }

        impl cmp::Eq for $name {}

        impl CurveExt for $name {

            type ScalarExt = $scalar;
            type Base = $base;
            type AffineExt = $name_affine;

            const CURVE_ID: &'static str = $curve_id;

            fn endo(&self) -> Self {
                self.endomorphism_base()
            }

            fn jacobian_coordinates(&self) -> ($base, $base, $base) {
                // Homogenous to Jacobian	
                let x = self.x * self.z;	
                let y = self.y * self.z.square();	
                (x, y, self.z)
            }


            fn hash_to_curve<'a>(_: &'a str) -> Box<dyn Fn(&[u8]) -> Self + 'a> {
                unimplemented!();
            }

            fn is_on_curve(&self) -> Choice {
                (self.z * self.y.square()  - self.x.square() * self.x - $name::curve_constant_a() * self.x * self.z.square())
                    .ct_eq(&(self.z.square() * self.z * $name::curve_constant_b()))
                    | self.z.is_zero()
            }

            fn b() -> Self::Base {
                $name::curve_constant_b()
            }

            fn a() -> Self::Base {
                $name::curve_constant_a()
            }

            fn new_jacobian(x: Self::Base, y: Self::Base, z: Self::Base) -> CtOption<Self> {
                // Jacobian to homogenous	
                let z_inv = z.invert().unwrap_or($base::zero());	
                let p_x = x * z_inv;	
                let p_y = y * z_inv.square();	
                let p = $name {	
                    x:p_x,	
                    y:$base::conditional_select(&p_y, &$base::one(), z.is_zero()),	
                    z	
                };
                CtOption::new(p, p.is_on_curve())
            }
        }

        impl group::Curve for $name {

            type AffineRepr = $name_affine;

            fn batch_normalize(p: &[Self], q: &mut [Self::AffineRepr]) {
                assert_eq!(p.len(), q.len());

                let mut acc = $base::one();
                for (p, q) in p.iter().zip(q.iter_mut()) {
                    // We use the `x` field of $name_affine to store the product
                    // of previous z-coordinates seen.
                    q.x = acc;

                    // We will end up skipping all identities in p
                    acc = $base::conditional_select(&(acc * p.z), &acc, p.is_identity());
                }

                // This is the inverse, as all z-coordinates are nonzero and the ones
                // that are not are skipped.
                acc = acc.invert().unwrap();

                for (p, q) in p.iter().rev().zip(q.iter_mut().rev()) {
                    let skip = p.is_identity();

                    // Compute tmp = 1/z
                    let tmp = q.x * acc;

                    // Cancel out z-coordinate in denominator of `acc`
                    acc = $base::conditional_select(&(acc * p.z), &acc, skip);

                    q.x = p.x * tmp;	
                    q.y = p.y * tmp;

                    *q = $name_affine::conditional_select(&q, &$name_affine::identity(), skip);
                }
            }

            fn to_affine(&self) -> Self::AffineRepr {
                let zinv = self.z.invert().unwrap_or($base::zero());	
                let x = self.x * zinv;	
                let y = self.y * zinv;

                let tmp = $name_affine {
                    x,
                    y,
                };

                $name_affine::conditional_select(&tmp, &$name_affine::identity(), zinv.is_zero())
            }
        }

        impl group::Group for $name {
            type Scalar = $scalar;

            fn random(mut rng: impl RngCore) -> Self {
                $name_affine::random(&mut rng).to_curve()
            }

            fn double(&self) -> Self {
                 // Algorithm 3, https://eprint.iacr.org/2015/1060.pdf
                let t0 = self.x.square();
                let t1 = self.y.square();
                let t2 = self.z.square();
                let t3 = self.x * self.y;
                let t3 = t3 + t3;
                let z3 = self.x * self.z;
                let z3 = z3 + z3;
                let x3 = $name::curve_constant_a() * z3;
                let y3 = $name::mul_by_3b(&t2);
                let y3 = x3 + y3;
                let x3 = t1 - y3;
                let y3 = t1 + y3;
                let y3 = x3 * y3;
                let x3 = t3 * x3;
                let z3 = $name::mul_by_3b(&z3);
                let t2 = $name::curve_constant_a() * t2;
                let t3 = t0 - t2;
                let t3 = $name::curve_constant_a() * t3;
                let t3 = t3 + z3;
                let z3 = t0 + t0;
                let t0 = z3 + t0;
                let t0 = t0 + t2;
                let t0 = t0 * t3;
                let y3 = y3 + t0;
                let t2 = self.y * self.z;
                let t2 = t2 + t2;
                let t0 = t2 * t3;
                let x3 = x3 - t0;
                let z3 = t2 * t1;
                let z3 = z3 + z3;
                let z3 = z3 + z3;

                let tmp = $name {
                    x: x3,
                    y: y3,
                    z: z3,
                };

                $name::conditional_select(&tmp, &$name::identity(), self.is_identity())
            }

            fn generator() -> Self {
                $name::generator()
            }

            fn identity() -> Self {
                Self {
                    x: $base::zero(),
                    y: $base::one(),
                    z: $base::zero(),
                }
            }

            fn is_identity(&self) -> Choice {
                self.z.is_zero()
            }
        }

        impl GroupEncoding for $name {
            type Repr = $name_compressed;

            fn from_bytes(bytes: &Self::Repr) -> CtOption<Self> {
                $name_affine::from_bytes(bytes).map(Self::from)
            }

            fn from_bytes_unchecked(bytes: &Self::Repr) -> CtOption<Self> {
                $name_affine::from_bytes(bytes).map(Self::from)
            }

            fn to_bytes(&self) -> Self::Repr {
                $name_affine::from(self).to_bytes()
            }
        }

        impl $crate::serde::SerdeObject for $name {
            fn from_raw_bytes_unchecked(bytes: &[u8]) -> Self {
                debug_assert_eq!(bytes.len(), 3 * $base::size());
                let [x, y, z] = [0, 1, 2]
                    .map(|i| $base::from_raw_bytes_unchecked(&bytes[i * $base::size()..(i + 1) * $base::size()]));
                Self { x, y, z }
            }
            fn from_raw_bytes(bytes: &[u8]) -> Option<Self> {
                if bytes.len() != 3 * $base::size() {
                    return None;
                }
                let [x, y, z] =
                    [0, 1, 2].map(|i| $base::from_raw_bytes(&bytes[i * $base::size()..(i + 1) * $base::size()]));
                x.zip(y).zip(z).and_then(|((x, y), z)| {
                    let res = Self { x, y, z };
                    // Check that the point is on the curve.
                    bool::from(res.is_on_curve()).then(|| res)
                })
            }
            fn to_raw_bytes(&self) -> Vec<u8> {
                let mut res = Vec::with_capacity(3 * $base::size());
                Self::write_raw(self, &mut res).unwrap();
                res
            }
            fn read_raw_unchecked<R: std::io::Read>(reader: &mut R) -> Self {
                let [x, y, z] = [(); 3].map(|_| $base::read_raw_unchecked(reader));
                Self { x, y, z }
            }
            fn read_raw<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
                let x = $base::read_raw(reader)?;
                let y = $base::read_raw(reader)?;
                let z = $base::read_raw(reader)?;
                Ok(Self { x, y, z })
            }
            fn write_raw<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
                self.x.write_raw(writer)?;
                self.y.write_raw(writer)?;
                self.z.write_raw(writer)
            }
        }

        impl group::prime::PrimeGroup for $name {}

        impl group::prime::PrimeCurve for $name {
            type Affine = $name_affine;
        }

        impl group::cofactor::CofactorCurve for $name {
            type Affine = $name_affine;
        }

        impl Group for $name {
            type Scalar = $scalar;

            fn group_zero() -> Self {
                Self::identity()
            }
            fn group_add(&mut self, rhs: &Self) {
                *self += *rhs;
            }
            fn group_sub(&mut self, rhs: &Self) {
                *self -= *rhs;
            }
            fn group_scale(&mut self, by: &Self::Scalar) {
                *self *= *by;
            }
        }

        // Affine implementations

        impl std::fmt::Debug for $name_affine {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
                if self.is_identity().into() {
                    write!(f, "Infinity")
                } else {
                    write!(f, "({:?}, {:?})", self.x, self.y)
                }
            }
        }

        impl<'a> From<&'a $name> for $name_affine {
            fn from(p: &'a $name) -> $name_affine {
                p.to_affine()
            }
        }

        impl From<$name> for $name_affine {
            fn from(p: $name) -> $name_affine {
                p.to_affine()
            }
        }

        impl Default for $name_affine {
            fn default() -> $name_affine {
                $name_affine::identity()
            }
        }

        impl subtle::ConstantTimeEq for $name_affine {
            fn ct_eq(&self, other: &Self) -> Choice {
                let z1 = self.is_identity();
                let z2 = other.is_identity();

                (z1 & z2) | ((!z1) & (!z2) & (self.x.ct_eq(&other.x)) & (self.y.ct_eq(&other.y)))
            }
        }

        impl subtle::ConditionallySelectable for $name_affine {
            fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
                $name_affine {
                    x: $base::conditional_select(&a.x, &b.x, choice),
                    y: $base::conditional_select(&a.y, &b.y, choice),
                }
            }
        }

        impl cmp::Eq for $name_affine {}

        impl group::GroupEncoding for $name_affine {
            type Repr = $name_compressed;

            fn from_bytes(bytes: &Self::Repr) -> CtOption<Self> {
                let bytes = &bytes.0;
                let mut tmp = *bytes;
                let ysign = Choice::from(tmp[$compressed_size - 1] >> 7);
                tmp[$compressed_size - 1] &= 0b0111_1111;
                let mut xbytes = [0u8; $base::size()];
                xbytes.copy_from_slice(&tmp[ ..$base::size()]);

                $base::from_bytes(&xbytes).and_then(|x| {
                    CtOption::new(Self::identity(), x.is_zero() & (!ysign)).or_else(|| {
                        let x3 = x.square() * x;
                        (x3 + $name::curve_constant_a() * x + $name::curve_constant_b()).sqrt().and_then(|y| {
                            let sign = Choice::from(y.to_bytes()[0] & 1);

                            let y = $base::conditional_select(&y, &-y, ysign ^ sign);

                            CtOption::new(
                                $name_affine {
                                    x,
                                    y,
                                },
                                Choice::from(1u8),
                            )
                        })
                    })
                })
            }

            fn from_bytes_unchecked(bytes: &Self::Repr) -> CtOption<Self> {
                Self::from_bytes(bytes)
            }

            fn to_bytes(&self) -> Self::Repr {
                if bool::from(self.is_identity()) {
                    $name_compressed::default()
                } else {
                    let (x, y) = (self.x, self.y);
                    let sign = (y.to_bytes()[0] & 1) << 7;
                    let mut xbytes = [0u8; $compressed_size];
                    xbytes[..$base::size()].copy_from_slice(&x.to_bytes());
                    xbytes[$compressed_size - 1] |= sign;
                    $name_compressed(xbytes)
                }
            }
        }

        impl $crate::serde::SerdeObject for $name_affine {
            fn from_raw_bytes_unchecked(bytes: &[u8]) -> Self {
                debug_assert_eq!(bytes.len(), 2 * $base::size());
                let [x, y] =
                    [0, $base::size()].map(|i| $base::from_raw_bytes_unchecked(&bytes[i..i + $base::size()]));
                Self { x, y }
            }
            fn from_raw_bytes(bytes: &[u8]) -> Option<Self> {
                if bytes.len() != 2 * $base::size() {
                    return None;
                }
                let [x, y] = [0, $base::size()].map(|i| $base::from_raw_bytes(&bytes[i..i + $base::size()]));
                x.zip(y).and_then(|(x, y)| {
                    let res = Self { x, y };
                    // Check that the point is on the curve.
                    bool::from(res.is_on_curve()).then(|| res)
                })
            }
            fn to_raw_bytes(&self) -> Vec<u8> {
                let mut res = Vec::with_capacity(2 * $base::size());
                Self::write_raw(self, &mut res).unwrap();
                res
            }
            fn read_raw_unchecked<R: std::io::Read>(reader: &mut R) -> Self {
                let [x, y] = [(); 2].map(|_| $base::read_raw_unchecked(reader));
                Self { x, y }
            }
            fn read_raw<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
                let x = $base::read_raw(reader)?;
                let y = $base::read_raw(reader)?;
                Ok(Self { x, y })
            }
            fn write_raw<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
                self.x.write_raw(writer)?;
                self.y.write_raw(writer)
            }
        }

        impl group::prime::PrimeCurveAffine for $name_affine {
            type Curve = $name;
            type Scalar = $scalar;


            fn generator() -> Self {
                $name_affine::generator()
            }

            fn identity() -> Self {
                Self {
                    x: $base::zero(),
                    y: $base::zero(),
                }
            }

            fn is_identity(&self) -> Choice {
                self.x.is_zero() & self.y.is_zero()
            }

            fn to_curve(&self) -> Self::Curve {
                let tmp = $name {	
                    x: self.x,	
                    y: self.y,	
                    z: $base::one(),	
                };	
                $name::conditional_select(&tmp, &$name::identity(), self.is_identity())
            }
        }

        impl group::cofactor::CofactorCurveAffine for $name_affine {
            type Curve = $name;
            type Scalar = $scalar;

            fn identity() -> Self {
                <Self as group::prime::PrimeCurveAffine>::identity()
            }

            fn generator() -> Self {
                <Self as group::prime::PrimeCurveAffine>::generator()
            }

            fn is_identity(&self) -> Choice {
                <Self as group::prime::PrimeCurveAffine>::is_identity(self)
            }

            fn to_curve(&self) -> Self::Curve {
                <Self as group::prime::PrimeCurveAffine>::to_curve(self)
            }
        }


        impl CurveAffine for $name_affine {
            type ScalarExt = $scalar;
            type Base = $base;
            type CurveExt = $name;

            fn is_on_curve(&self) -> Choice {
               // y^2 - x^3 - ax ?= b
               (self.y.square() - self.x.square() * self.x - $name::curve_constant_a() * self.x).ct_eq(&$name::curve_constant_b())
               | self.is_identity()
            }

            fn coordinates(&self) -> CtOption<Coordinates<Self>> {
                Coordinates::from_xy( self.x, self.y )
            }

            fn from_xy(x: Self::Base, y: Self::Base) -> CtOption<Self> {
                let p = $name_affine {
                    x, y
                };
                CtOption::new(p, p.is_on_curve())
            }

            fn a() -> Self::Base {
                $name::curve_constant_a()
            }

            fn b() -> Self::Base {
                $name::curve_constant_b()
            }
        }


        impl_binops_additive!($name, $name);
        impl_binops_additive!($name, $name_affine);
        impl_binops_additive_specify_output!($name_affine, $name_affine, $name);
        impl_binops_additive_specify_output!($name_affine, $name, $name);
        impl_binops_multiplicative!($name, $scalar);
        impl_binops_multiplicative_mixed!($name_affine, $scalar, $name);

        impl<'a> Neg for &'a $name {
            type Output = $name;

            fn neg(self) -> $name {
                $name {
                    x: self.x,
                    y: -self.y,
                    z: self.z,
                }
            }
        }

        impl Neg for $name {
            type Output = $name;

            fn neg(self) -> $name {
                -&self
            }
        }

        impl<T> Sum<T> for $name
        where
            T: core::borrow::Borrow<$name>,
        {
            fn sum<I>(iter: I) -> Self
            where
                I: Iterator<Item = T>,
            {
                iter.fold(Self::identity(), |acc, item| acc + item.borrow())
            }
        }

        impl<'a, 'b> Add<&'a $name> for &'b $name {
            type Output = $name;

            fn add(self, rhs: &'a $name) -> $name {
                 // Algorithm 1, https://eprint.iacr.org/2015/1060.pdf
                 let t0 = self.x * rhs.x;
                 let t1 = self.y * rhs.y;
                 let t2 = self.z * rhs.z;
                 let t3 = self.x + self.y;
                 let t4 = rhs.x + rhs.y;
                 let t3 = t3 * t4;
                 let t4 = t0 + t1;
                 let t3 = t3 - t4;
                 let t4 = self.x + self.z;
                 let t5 = rhs.x + rhs.z;
                 let t4 = t4 * t5;
                 let t5 = t0 + t2;
                 let t4 = t4 - t5;
                 let t5 = self.y + self.z;
                 let x3 = rhs.y + rhs.z;
                 let t5 = t5 * x3;
                 let x3 = t1 + t2;
                 let t5 = t5 - x3;
                 let z3 = $name::curve_constant_a() * t4;
                 let x3 = $name::mul_by_3b(&t2);
                 let z3 = x3 + z3;
                 let x3 = t1 - z3;
                 let z3 = t1 + z3;
                 let y3 = x3 * z3;
                 let t1 = t0 + t0;
                 let t1 = t1 + t0;
                 let t2 = $name::curve_constant_a() * t2;
                 let t4 = $name::mul_by_3b(&t4);
                 let t1 = t1 + t2;
                 let t2 = t0 - t2;
                 let t2 = $name::curve_constant_a() * t2;
                 let t4 = t4 + t2;
                 let t0 = t1 * t4;
                 let y3 = y3 + t0;
                 let t0 = t5 * t4;
                 let x3 = t3 * x3;
                 let x3 = x3 - t0;
                 let t0 = t3 * t1;
                 let z3 = t5 * z3;
                 let z3 = z3 + t0;
 
                 $name {
                     x: x3,
                     y: y3,
                     z: z3,
                 }


            }
        }

        impl<'a, 'b> Add<&'a $name_affine> for &'b $name {
            type Output = $name;

            fn add(self, rhs: &'a $name_affine) -> $name {
                // Algorithm 2, https://eprint.iacr.org/2015/1060.pdf
                let t0 = self.x * rhs.x;
                let t1 = self.y * rhs.y;
                let t3 = rhs.x + rhs.y;
                let t4 = self.x + self.y;
                let t3 = t3 * t4;
                let t4 = t0 + t1;
                let t3 = t3 - t4;
                let t4 = rhs.x * self.z;
                let t4 = t4 + self.x;
                let t5 = rhs.y * self.z;
                let t5 = t5 + self.y;
                let z3 = $name::curve_constant_a() * t4;
                let x3 = $name::mul_by_3b(&self.z);
                let z3 = x3 + z3;
                let x3 = t1 - z3;
                let z3 = t1 + z3;
                let y3 = x3 * z3;
                let t1 = t0 + t0;
                let t1 = t1 + t0;
                let t2 = $name::curve_constant_a() * self.z;
                let t4 = $name::mul_by_3b(&t4);
                let t1 = t1 + t2;
                let t2 = t0 - t2;
                let t2 = $name::curve_constant_a() * t2;
                let t4 = t4 + t2;
                let t0 = t1 * t4;
                let y3 = y3 + t0;
                let t0 = t5 * t4;
                let x3 = t3 * x3;
                let x3 = x3 - t0;
                let t0 = t3 * t1;
                let z3 = t5 * z3;
                let z3 = z3 + t0;

                let tmp = $name{
                    x: x3,
                    y: y3,
                    z: z3,
                };

                $name::conditional_select(&tmp, self, rhs.is_identity())
            }
        }

        impl<'a, 'b> Sub<&'a $name> for &'b $name {
            type Output = $name;

            fn sub(self, other: &'a $name) -> $name {
                self + (-other)
            }
        }

        impl<'a, 'b> Sub<&'a $name_affine> for &'b $name {
            type Output = $name;

            fn sub(self, other: &'a $name_affine) -> $name {
                self + (-other)
            }
        }



        #[allow(clippy::suspicious_arithmetic_impl)]
        impl<'a, 'b> Mul<&'b $scalar> for &'a $name {
            type Output = $name;

            // This is a simple double-and-add implementation of point
            // multiplication, moving from most significant to least
            // significant bit of the scalar.

            fn mul(self, other: &'b $scalar) -> Self::Output {
                let mut acc = $name::identity();
                for bit in other
                    .to_repr()
                    .iter()
                    .rev()
                    .flat_map(|byte| (0..8).rev().map(move |i| Choice::from((byte >> i) & 1u8)))
                {
                    acc = acc.double();
                    acc = $name::conditional_select(&acc, &(acc + self), bit);
                }

                acc
            }
        }

        impl<'a> Neg for &'a $name_affine {
            type Output = $name_affine;

            fn neg(self) -> $name_affine {
                $name_affine {
                    x: self.x,
                    y: -self.y,
                }
            }
        }

        impl Neg for $name_affine {
            type Output = $name_affine;

            fn neg(self) -> $name_affine {
                -&self
            }
        }

        impl<'a, 'b> Add<&'a $name> for &'b $name_affine {
            type Output = $name;

            fn add(self, rhs: &'a $name) -> $name {
                rhs + self
            }
        }

        impl<'a, 'b> Add<&'a $name_affine> for &'b $name_affine {
            type Output = $name;

            fn add(self, rhs: &'a $name_affine) -> $name {
                rhs.to_curve() + self.to_curve()
            }
        }

        impl<'a, 'b> Sub<&'a $name_affine> for &'b $name_affine {
            type Output = $name;

            fn sub(self, other: &'a $name_affine) -> $name {
                self + (-other)
            }
        }

        impl<'a, 'b> Sub<&'a $name> for &'b $name_affine {
            type Output = $name;

            fn sub(self, other: &'a $name) -> $name {
                self + (-other)
            }
        }

        #[allow(clippy::suspicious_arithmetic_impl)]
        impl<'a, 'b> Mul<&'b $scalar> for &'a $name_affine {
            type Output = $name;

            fn mul(self, other: &'b $scalar) -> Self::Output {
                let mut acc = $name::identity();

                // This is a simple double-and-add implementation of point
                // multiplication, moving from most significant to least
                // significant bit of the scalar.

                for bit in other
                    .to_repr()
                    .iter()
                    .rev()
                    .flat_map(|byte| (0..8).rev().map(move |i| Choice::from((byte >> i) & 1u8)))
                {
                    acc = acc.double();
                    acc = $name::conditional_select(&acc, &(acc + self), bit);
                }

                acc
            }
        }
    };
}
