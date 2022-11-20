use std::marker::PhantomData;

/// An `m` by `n` matrix i.e. `m` rows and `n` columns.
/// Stored in row-major order.
pub struct Matrix<T, const M: usize, const N: usize, const S: usize> {
    pub inner: Box<[T; S]>,
    _width: PhantomData<[u8; N]>,
    _height: PhantomData<[u8; M]>,
}

impl<T, const M: usize, const N: usize, const S: usize> Matrix<T, M, N, S>
where
    T: Default + Copy,
{
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.inner.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.inner.iter_mut()
    }

    pub fn rows(&self) -> impl Iterator<Item = &[T]> {
        self.inner.chunks_exact(N)
    }

    pub fn transpose(&self) -> Matrix<T, N, M, S> {
        let mut transpose = Matrix::default();

        fn index(row: usize, col: usize, num_cols: usize) -> usize {
            row * num_cols + col
        }

        for row in 0..N {
            for col in 0..M {
                let new_index = index(row, col, M);
                let old_index = index(col, row, N);
                transpose.inner[new_index] = self.inner[old_index];
            }
        }

        transpose
    }
}

fn vec_to_boxed_array<T: Copy, const S: usize>(val: T) -> Box<[T; S]> {
    let boxed_slice = vec![val; S].into_boxed_slice();

    let ptr = Box::into_raw(boxed_slice) as *mut [T; S];

    unsafe { Box::from_raw(ptr) }
}

impl<T: Default + Copy, const M: usize, const N: usize, const S: usize> Default
    for Matrix<T, M, N, S>
{
    fn default() -> Self {
        assert_eq!(M * N, S);
        Self {
            inner: vec_to_boxed_array(T::default()),
            _width: PhantomData,
            _height: PhantomData,
        }
    }
}
