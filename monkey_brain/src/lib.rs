mod load_mnist;
mod net;

pub use load_mnist::{MnistImage, load_data, one_hot};
pub use net::{Net, SGD};