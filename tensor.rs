// Rust implementations of the morloc tensor stdlib. Matrix/Tensor* map to
// ndarray's fixed-rank arrays (Array2..Array5). ndarray is row-major by
// default, matching morloc's row-major wire form, so marshalling is a flat
// reshape with no transpose.
//
// Argument-passing follows the morloc Rust FFI convention: Copy scalars by
// value, non-Copy values and generics by shared reference.

use ndarray::{Array2, Array3, Array4, Array5, LinalgScalar};

// Convert a wire/user dimension to usize. A negative dimension would wrap to a
// near-2^64 usize under `as`, turning a bad input into an OOM/capacity-overflow
// abort; throw a catchable morloc error instead (matching numpy's ValueError).
fn usz(x: i64) -> usize {
    if x < 0 {
        rustmorloc::morloc_throw("tensor dimension must be non-negative");
    }
    x as usize
}

// 0/1 per numeric type, for the result-typed zeros/ones/identity constructors.
// Self-contained (a distinct name from vector-rust's equivalent trait): morloc
// includes a sourced .rs only when its symbols are USED, so vector.rs is not
// guaranteed to be in a tensor-only pool -- tensor.rs must stand alone.
trait MorlocTensorNum {
    fn m_zero() -> Self;
    fn m_one() -> Self;
}
macro_rules! morloc_tensor_num {
    ($($t:ty),*) => {
        $(impl MorlocTensorNum for $t {
            fn m_zero() -> Self { 0 as $t }
            fn m_one() -> Self { 1 as $t }
        })*
    };
}
morloc_tensor_num!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

// Construction. zeros/ones take the fill value from a MorlocTensorNum producer;
// fill takes an explicit value by reference. from_shape_vec reads the flat vec
// in row-major order. The error string is derived from the function name so it
// cannot drift.
macro_rules! tensor_ctor {
    // zeros/ones: fill value is A::$producer() (a MorlocTensorNum method). The
    // method name is passed (not `A::m_zero()`) so the type parameter A stays
    // internal to the generated function (macro hygiene).
    (num $name:ident, $arr:ident, ($($d:ident),+), $producer:ident) => {
        pub fn $name<A: MorlocTensorNum + Clone>($($d: i64),+) -> $arr<A> {
            let n = 1usize $(* usz($d))+;
            $arr::from_shape_vec(($(usz($d)),+), vec![A::$producer(); n])
                .unwrap_or_else(|_| rustmorloc::morloc_throw(concat!(stringify!($name), ": shape error")))
        }
    };
    (fill $name:ident, $arr:ident, ($($d:ident),+)) => {
        pub fn $name<A: Clone>(v: &A, $($d: i64),+) -> $arr<A> {
            let n = 1usize $(* usz($d))+;
            $arr::from_shape_vec(($(usz($d)),+), vec![v.clone(); n])
                .unwrap_or_else(|_| rustmorloc::morloc_throw(concat!(stringify!($name), ": shape error")))
        }
    };
}

tensor_ctor!(num morloc_zeros2, Array2, (d1, d2), m_zero);
tensor_ctor!(num morloc_zeros3, Array3, (d1, d2, d3), m_zero);
tensor_ctor!(num morloc_zeros4, Array4, (d1, d2, d3, d4), m_zero);
tensor_ctor!(num morloc_zeros5, Array5, (d1, d2, d3, d4, d5), m_zero);
tensor_ctor!(num morloc_ones2, Array2, (d1, d2), m_one);
tensor_ctor!(num morloc_ones3, Array3, (d1, d2, d3), m_one);
tensor_ctor!(num morloc_ones4, Array4, (d1, d2, d3, d4), m_one);
tensor_ctor!(num morloc_ones5, Array5, (d1, d2, d3, d4, d5), m_one);
tensor_ctor!(fill morloc_fill2, Array2, (d1, d2));
tensor_ctor!(fill morloc_fill3, Array3, (d1, d2, d3));
tensor_ctor!(fill morloc_fill4, Array4, (d1, d2, d3, d4));
tensor_ctor!(fill morloc_fill5, Array5, (d1, d2, d3, d4, d5));

pub fn morloc_identity<A: MorlocTensorNum + Clone>(n: i64) -> Array2<A> {
    let m = usz(n);
    let mut data = vec![A::m_zero(); m * m];
    for i in 0..m {
        data[i * m + i] = A::m_one();
    }
    Array2::from_shape_vec((m, m), data)
        .unwrap_or_else(|_| rustmorloc::morloc_throw("identity: shape error"))
}

pub fn morloc_matmul<A: LinalgScalar>(a: &Array2<A>, b: &Array2<A>) -> Array2<A> {
    // ndarray's 2-D dot panics on a dimension mismatch; surface it as a
    // catchable morloc error instead (the type-level `k` is not guaranteed to
    // match the runtime shape for gradual dims or externally-built matrices).
    if a.ncols() != b.nrows() {
        rustmorloc::morloc_throw("matmul: inner dimensions do not match");
    }
    a.dot(b)
}

// Structural equality across ranks (ndarray's Array implements PartialEq
// element-wise, comparing shape and contents).
pub fn morloc_tensor_eq<A: PartialEq, D: ndarray::Dimension>(
    a: &ndarray::Array<A, D>,
    b: &ndarray::Array<A, D>,
) -> bool {
    a == b
}

// ---------------------------------------------------------------------------
// Packable: wire form ((dims...), flat Vec, row-major) <-> shaped ArrayN.
// `.iter()` on any ndarray Array yields elements in logical (row-major) order,
// so unpack produces the row-major flat buffer the wire form expects, and
// from_shape_vec reads it back in the same order. The clones are required by the
// by-reference FFI convention (from_shape_vec needs an owned Vec; one cannot
// move out of &Array).
// ---------------------------------------------------------------------------

pub fn morloc_packMatrix<A: Clone>(packed: &((i64, i64), Vec<A>)) -> Array2<A> {
    let ((d1, d2), data) = packed;
    Array2::from_shape_vec((usz(*d1), usz(*d2)), data.clone())
        .unwrap_or_else(|_| rustmorloc::morloc_throw("packMatrix: shape/length mismatch"))
}

pub fn morloc_unpackMatrix<A: Clone>(t: &Array2<A>) -> ((i64, i64), Vec<A>) {
    let s = t.shape();
    ((s[0] as i64, s[1] as i64), t.iter().cloned().collect())
}

pub fn morloc_packTensor3<A: Clone>(packed: &((i64, i64, i64), Vec<A>)) -> Array3<A> {
    let ((d1, d2, d3), data) = packed;
    Array3::from_shape_vec((usz(*d1), usz(*d2), usz(*d3)), data.clone())
        .unwrap_or_else(|_| rustmorloc::morloc_throw("packTensor3: shape/length mismatch"))
}

pub fn morloc_unpackTensor3<A: Clone>(t: &Array3<A>) -> ((i64, i64, i64), Vec<A>) {
    let s = t.shape();
    (
        (s[0] as i64, s[1] as i64, s[2] as i64),
        t.iter().cloned().collect(),
    )
}

pub fn morloc_packTensor4<A: Clone>(packed: &((i64, i64, i64, i64), Vec<A>)) -> Array4<A> {
    let ((d1, d2, d3, d4), data) = packed;
    Array4::from_shape_vec((usz(*d1), usz(*d2), usz(*d3), usz(*d4)), data.clone())
        .unwrap_or_else(|_| rustmorloc::morloc_throw("packTensor4: shape/length mismatch"))
}

pub fn morloc_unpackTensor4<A: Clone>(t: &Array4<A>) -> ((i64, i64, i64, i64), Vec<A>) {
    let s = t.shape();
    (
        (s[0] as i64, s[1] as i64, s[2] as i64, s[3] as i64),
        t.iter().cloned().collect(),
    )
}

pub fn morloc_packTensor5<A: Clone>(packed: &((i64, i64, i64, i64, i64), Vec<A>)) -> Array5<A> {
    let ((d1, d2, d3, d4, d5), data) = packed;
    Array5::from_shape_vec(
        (usz(*d1), usz(*d2), usz(*d3), usz(*d4), usz(*d5)),
        data.clone(),
    )
    .unwrap_or_else(|_| rustmorloc::morloc_throw("packTensor5: shape/length mismatch"))
}

pub fn morloc_unpackTensor5<A: Clone>(t: &Array5<A>) -> ((i64, i64, i64, i64, i64), Vec<A>) {
    let s = t.shape();
    (
        (
            s[0] as i64,
            s[1] as i64,
            s[2] as i64,
            s[3] as i64,
            s[4] as i64,
        ),
        t.iter().cloned().collect(),
    )
}
