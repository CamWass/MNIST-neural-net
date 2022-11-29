use data::{as_chunks, encode_image};
use network::{NeuralNet as NN, State};

use postcard::from_bytes;
use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

static RAW_STATE: &[u8] = include_bytes!("../../../pretrained/pretrained-96");

fn make_neural_net() -> NN {
    let state: State = from_bytes(RAW_STATE).unwrap();
    NN::restore_from_state(state)
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

    pub fn classify(&mut self, image: &[u8], breakdown: &mut [f32]) -> u8 {
        let image = as_chunks(image);
        assert_eq!(image.0.len(), 1);
        assert_eq!(image.1.len(), 0);
        let image_data = &image.0[0];
        let mut image = Box::new([0.0; 784]);
        encode_image(image_data, &mut image);

        let classification = self.0.classify(&image);
        breakdown.copy_from_slice(classification);

        let prediction = classification
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap() as u8;
        prediction
    }
}
