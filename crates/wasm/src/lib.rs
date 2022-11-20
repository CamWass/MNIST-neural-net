use data::{as_chunks, encode_image};
use network::{NeuralNet as NN, Params};

use postcard::from_bytes;
use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

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
        let image_data = &image.0[0];
        let mut image = Box::new([0.0; 784]);
        encode_image(image_data, &mut image);
        let prediction = self.0.classify(&image);
        prediction
    }
}
