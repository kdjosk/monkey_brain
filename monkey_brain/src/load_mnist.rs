use std::{fs::File, io::{Read, Cursor}};
use byteorder::{BigEndian, ReadBytesExt};
use ndarray::Array2; 


pub fn load_data(dataset_name: &str) -> Result<Vec<MnistImage>, std::io::Error> {
    let filename = format!("dataset/{}-labels-idx1-ubyte.gz", dataset_name);
    let label_data = MnistData::from_file(&File::open(filename)?)?;
    let filename = format!("dataset/{}-images-idx3-ubyte.gz", dataset_name);
    let images_data = MnistData::from_file(&File::open(filename)?)?;
    let mut images: Vec<Array2<f64>> = Vec::new();
    let pixel_count = (images_data.sizes[1] * images_data.sizes[2]) as usize;

    for i in 0..images_data.sizes[0] as usize {
        let start = i * pixel_count;
        let image_data = images_data.data[start..start + pixel_count].to_vec();
        let image_data: Vec<f64> = image_data.into_iter().map(|x| x as f64 / 255.).collect();
        images.push(Array2::from_shape_vec((pixel_count, 1), image_data).unwrap());
    }

    let labels: Vec<u8> = label_data.data.clone();
    let mut ret: Vec<MnistImage> = Vec::new();

    for (image, label) in images.into_iter().zip(labels) {
        ret.push(MnistImage {
            image,
            label,
        })
}

    Ok(ret)
}

pub fn one_hot(label: usize) -> Array2<f64> {
    let mut a = Array2::zeros((10, 1));
    a[[label, 0]] = 1.0;
    a
}

#[derive(Debug)]
pub struct MnistImage {
    pub image: Array2<f64>,
    pub label: u8,
}

#[derive(Debug)]
struct MnistData {
    sizes: Vec<i32>,
    data: Vec<u8>,
}

impl MnistData {
    fn from_file(f: &File) -> Result<MnistData, std::io::Error> {
        let mut gz = flate2::read::GzDecoder::new(f);
        let mut contents: Vec<u8> = Vec::new();
        gz.read_to_end(&mut contents)?;
        let mut r = Cursor::new(&contents);

        let magic_number = r.read_i32::<BigEndian>()?;

        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        match magic_number {
            2049 => {
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            _ => panic!(),
        }
        r.read_to_end(&mut data)?;

        Ok(MnistData { sizes, data })
    }
}

// label file
// [offset] [type]          [value]          [description]
// 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
// 0004     32 bit integer  60000            number of items
// 0008     unsigned byte   ??               label
// 0009     unsigned byte   ??               label
// ........
// xxxx     unsigned byte   ??               label

// The labels values are 0 to 9. 


// image file
// [offset] [type]          [value]          [description]
// 0000     32 bit integer  0x00000803(2051) magic number
// 0004     32 bit integer  60000            number of images
// 0008     32 bit integer  28               number of rows
// 0012     32 bit integer  28               number of columns
// 0016     unsigned byte   ??               pixel
// 0017     unsigned byte   ??               pixel
// ........
// xxxx     unsigned byte   ??               pixel