use network::{NeuralNet as NN, Params};

use postcard::from_bytes;
use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

// TODO: deduplicate with cli
fn encode_image(image: &[u8; 784]) -> Box<[f32; 784]> {
    let mut a = Box::new([0.0; 784]);
    for (i, &pixel) in image.iter().enumerate() {
        a[i] = (pixel as f32) / 255.0;
    }
    a
}

fn as_chunks<T, const N: usize>(slice: &[T]) -> (&[[T; N]], &[T]) {
    assert_ne!(N, 0);
    let len = slice.len() / N;
    let (multiple_of_n, remainder) = slice.split_at(len * N);
    // SAFETY: We already panicked for zero, and ensured by construction
    // that the length of the subslice is a multiple of N.
    let array_slice = unsafe {
        // Inlined `as_chunks_unchecked` from nightly std

        // SAFETY: Caller must guarantee that `N` is nonzero and exactly divides the slice length
        let new_len = len;
        // SAFETY: We cast a slice of `new_len * N` elements into
        // a slice of `new_len` many `N` elements chunks.
        std::slice::from_raw_parts(multiple_of_n.as_ptr().cast(), new_len)
    };
    (array_slice, remainder)
}

static RAW_PARAMS: &[u8] = include_bytes!("../../../pretrained/pretrained-params-81");

fn make_neural_net() -> NN {
    let params: Box<Params> = from_bytes(RAW_PARAMS).unwrap();
    NN::from_params(params)
}

#[wasm_bindgen]
pub struct NeuralNet(NN);

#[wasm_bindgen]
impl NeuralNet {
    pub fn new() -> NeuralNet {
        // When the `console_error_panic_hook` feature is enabled, we can call the
        // `set_panic_hook` function at least once during initialization, and then
        // we will get better error messages if our code ever panics.
        //
        // For more details see
        // https://github.com/rustwasm/console_error_panic_hook#readme
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();

        NeuralNet(make_neural_net())
    }

    pub fn classify(&mut self, image: &[u8]) -> u8 {
        let image = as_chunks(image);
        assert_eq!(image.0.len(), 1);
        assert_eq!(image.1.len(), 0);
        let image = &image.0[0];
        let image = encode_image(image);
        let prediction = self.0.classify(&image);
        prediction
    }
}
