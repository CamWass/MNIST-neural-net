use std::{fmt, marker::PhantomData};

use serde::{
    de::{self, SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};

struct ArrayVisitor<T, const S: usize> {
    _marker: PhantomData<T>,
}

impl<'de, T, const S: usize> Visitor<'de> for ArrayVisitor<T, S>
where
    T: Deserialize<'de> + Default + Copy,
{
    type Value = Box<[T; S]>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "an array of size {}", S)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut arr = vec_to_boxed_array(T::default());

        let mut i = 0;
        while let Some(val) = seq.next_element()? {
            arr[i] = val;
            i += 1;
        }

        if i != S {
            Err(de::Error::invalid_length(arr.len(), &self))
        } else {
            Ok(arr)
        }
    }
}

pub fn deserialize<'de, D, T, const S: usize>(deserialize: D) -> Result<Box<[T; S]>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + Default + Copy,
{
    deserialize.deserialize_tuple(
        S,
        ArrayVisitor {
            _marker: PhantomData,
        },
    )
}

/// An `m` by `n` matrix i.e. `m` rows and `n` columns.
/// Stored in row-major order.
#[derive(Serialize, Deserialize)]
pub struct Matrix<T, const M: usize, const N: usize, const S: usize> {
    #[serde(serialize_with = "serde_arrays::serialize")]
    #[serde(deserialize_with = "deserialize")]
    #[serde(bound(serialize = "T: Serialize"))]
    #[serde(bound(deserialize = "T: Deserialize<'de> + Default + Copy"))]
    pub inner: Box<[T; S]>,
    _width: PhantomData<[u8; N]>,
    _height: PhantomData<[u8; M]>,
}

impl<T, const M: usize, const N: usize, const S: usize> Matrix<T, M, N, S> {
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.inner.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.inner.iter_mut()
    }

    pub fn rows(&self) -> impl Iterator<Item = &[T]> {
        self.inner.chunks_exact(N)
    }

    pub fn transpose(&self) -> TransposedMatrix<T, N, M, S> {
        TransposedMatrix { inner: self }
    }
}

// TODO: may be able to replace this with somthing safe using .try_into()
fn vec_to_boxed_array<T: Copy, const S: usize>(val: T) -> Box<[T; S]> {
    let boxed_slice = vec![val; S].into_boxed_slice();

    let ptr = Box::into_raw(boxed_slice) as *mut [T; S];

    unsafe { Box::from_raw(ptr) }
}

impl<T, const M: usize, const N: usize, const S: usize> Default for Matrix<T, M, N, S>
where
    T: Default + Copy + Serialize,
{
    fn default() -> Self {
        debug_assert_eq!(M * N, S);
        Self {
            inner: vec_to_boxed_array(T::default()),
            _width: PhantomData,
            _height: PhantomData,
        }
    }
}

pub struct TransposedMatrix<'a, T, const M: usize, const N: usize, const S: usize> {
    inner: &'a Matrix<T, N, M, S>,
}

pub trait MatrixMultiply<T, const M: usize, const N: usize> {
    fn multiply_by(&self, other: &[T; N]) -> [T; M];
}

impl<const M: usize, const N: usize, const S: usize> MatrixMultiply<f32, M, N>
    for Matrix<f32, M, N, S>
{
    fn multiply_by(&self, other: &[f32; N]) -> [f32; M] {
        let mut result = [0.0; M];

        for (i, row) in self.rows().enumerate() {
            result[i] = row.iter().zip(other).map(|(a, b)| a * b).sum();
        }

        result
    }
}

impl<'a, const M: usize, const N: usize, const S: usize> MatrixMultiply<f32, M, N>
    for TransposedMatrix<'a, f32, M, N, S>
{
    fn multiply_by(&self, other: &[f32; N]) -> [f32; M] {
        let mut result = [0.0; M];

        for col in 0..M {
            let mut r = 0.0;
            for row in 0..N {
                r += self.inner.inner[row * M + col] * other[row];
            }
            result[col] = r;
        }

        result
    }
}
