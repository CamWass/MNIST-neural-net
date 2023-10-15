use std::{
    fmt,
    marker::PhantomData,
    ops::{AddAssign, Mul},
};

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
        let mut arr = vec_to_boxed_array(vec![T::default(); S]);

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

    pub fn transpose(&self) -> TransposedMatrix<T, N, M, S> {
        TransposedMatrix { inner: self }
    }
}

fn vec_to_boxed_array<T, const S: usize>(vec: Vec<T>) -> Box<[T; S]> {
    match vec.try_into() {
        Ok(v) => v,
        Err(_) => {
            unreachable!("vec has unexpected size");
        }
    }
}

impl<T, const M: usize, const N: usize, const S: usize> Matrix<T, M, N, S>
where
    T: Default + Copy,
{
    fn from_vec(vec: Vec<T>) -> Self {
        Self {
            inner: vec_to_boxed_array(vec),
            _width: PhantomData,
            _height: PhantomData,
        }
    }
}

impl<T, const M: usize, const N: usize, const S: usize> Default for Matrix<T, M, N, S>
where
    T: Default + Copy,
{
    fn default() -> Self {
        debug_assert_eq!(M * N, S);
        Self::from_vec(vec![T::default(); S])
    }
}

pub struct TransposedMatrix<'a, T, const M: usize, const N: usize, const S: usize> {
    inner: &'a Matrix<T, N, M, S>,
}

const BLOCK_SIZE: usize = 8;

fn matrix_multiply<T, const M: usize, const N: usize>(
    matrix: &[T],
    other: &[T; N],
    transpose: bool,
) -> [T; M]
where
    T: Mul<Output = T> + Default + Copy + AddAssign,
{
    let mut result = [T::default(); M];

    let mut i = 0;
    while i < M {
        let mut j = 0;
        while j < N {
            let mut x = i;
            while x < M.min(i + BLOCK_SIZE) {
                let mut y = j;
                while y < N.min(j + BLOCK_SIZE) {
                    if transpose {
                        result[x] += matrix[y * M + x] * other[y];
                    } else {
                        result[x] += matrix[x * N + y] * other[y];
                    }

                    y += 1;
                }
                x += 1;
            }
            j += BLOCK_SIZE;
        }
        i += BLOCK_SIZE;
    }
    result
}

pub trait MatrixMultiply<T, const M: usize, const N: usize> {
    fn multiply_by(&self, other: &[T; N]) -> [T; M];
}

impl<T, const M: usize, const N: usize, const S: usize> MatrixMultiply<T, M, N>
    for Matrix<T, M, N, S>
where
    T: Mul<Output = T> + Default + Copy + AddAssign,
{
    fn multiply_by(&self, other: &[T; N]) -> [T; M] {
        matrix_multiply(self.inner.as_ref(), other, false)
    }
}

impl<'a, T, const M: usize, const N: usize, const S: usize> MatrixMultiply<T, M, N>
    for TransposedMatrix<'a, T, M, N, S>
where
    T: Mul<Output = T> + Default + Copy + AddAssign,
{
    fn multiply_by(&self, other: &[T; N]) -> [T; M] {
        matrix_multiply(self.inner.inner.as_ref(), other, true)
    }
}

#[test]
fn test_square_matrix_multiplication() {
    let matrix: Matrix<u32, 3, 3, 9> = Matrix::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let vec = vec_to_boxed_array(vec![97, 98, 99]);
    let result = matrix.multiply_by(&vec);
    assert_eq!(result, [590, 1472, 2354]);
}

#[test]
fn test_rectangle_matrix_multiplication() {
    let matrix: Matrix<u32, 4, 3, 12> =
        Matrix::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let vec = vec_to_boxed_array(vec![97, 98, 99]);
    let result = matrix.multiply_by(&vec);
    assert_eq!(result, [590, 1472, 2354, 3236]);
}

#[test]
fn test_square_transposed_matrix_multiplication() {
    let matrix: Matrix<u32, 3, 3, 9> = Matrix::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let vec = vec_to_boxed_array(vec![97, 98, 99]);
    let result = matrix.transpose().multiply_by(&vec);
    assert_eq!(result, [1182, 1476, 1770]);
}

#[test]
fn test_rectangle_transposed_matrix_multiplication() {
    let matrix: Matrix<u32, 4, 3, 12> =
        Matrix::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let vec = vec_to_boxed_array(vec![96, 97, 98, 99]);
    let result = matrix.transpose().multiply_by(&vec);
    assert_eq!(result, [2160, 2550, 2940]);
}
