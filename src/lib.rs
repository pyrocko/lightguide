use ndarray::prelude::*;
use std::collections::HashMap;

use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use fftw::array::AlignedVec;
use fftw::plan::*;
use fftw::types::*;

use rayon;
use rayon::prelude::*;

use std::cmp;

#[derive(Hash)]
struct Tile {
    px_x: (usize, usize),
    px_y: (usize, usize),
    ix: usize,
    iy: usize,
}

impl PartialEq for Tile {
    fn eq(&self, other: &Self) -> bool {
        self.ix == other.ix && self.iy == other.iy
    }
}

impl Eq for Tile {}

fn triangular_taper(window_size: usize, plateau: usize) -> Array2<f32> {
    assert!(plateau < window_size, "Plateau cannot be larger than size.");
    assert!(window_size as f32 % 2.0 == 0., "Size has to be even");

    let ramp_size = (window_size - plateau) / 2;
    // Crop 0.0 and 1.0
    let ramp = Array1::<f32>::linspace(0., 1., ramp_size + 2)
        .slice(s![1..-1])
        .to_owned();

    let mut taper = Array1::<f32>::ones(window_size);
    taper.slice_mut(s![..ramp_size]).assign(&ramp);
    taper
        .slice_mut(s![window_size - ramp_size..])
        .assign(&ramp.slice(s![..;-1]));

    let taper = &taper * &taper.slice(s![.., NewAxis]);
    taper
}

fn triangular_taper_2d(window_size: (usize, usize), plateau: (usize, usize)) -> Array2<f32> {
    assert!(
        plateau.0 < window_size.0 && plateau.1 < window_size.1,
        "Plateau cannot be larger than size."
    );
    assert!(
        window_size.0 as f32 % 2.0 == 0. && window_size.1 as f32 % 2.0 == 0.,
        "Window size has to be even"
    );

    let ramp_size = (
        (window_size.0 - plateau.0) / 2,
        (window_size.1 - plateau.1) / 2,
    );
    // Crop 0.0 and 1.0
    let ramp = (
        Array1::<f32>::linspace(0., 1., ramp_size.0 + 2)
            .slice(s![1..-1])
            .to_owned(),
        Array1::<f32>::linspace(0., 1., ramp_size.1 + 2)
            .slice(s![1..-1])
            .to_owned(),
    );

    let mut taper = (
        Array1::<f32>::ones(window_size.0).to_owned(),
        Array1::<f32>::ones(window_size.1).to_owned(),
    );

    taper.0.slice_mut(s![..ramp_size.0]).assign(&ramp.0);
    taper
        .0
        .slice_mut(s![window_size.0 - ramp_size.0..])
        .assign(&ramp.0.slice(s![..;-1]));

    taper.1.slice_mut(s![..ramp_size.1]).assign(&ramp.1);
    taper
        .1
        .slice_mut(s![window_size.1 - ramp_size.1..])
        .assign(&ramp.1.slice(s![..;-1]));

    let window = &taper.1 * &taper.0.slice(s![.., NewAxis]);
    window
}

fn afk_filter_rectangular(
    data: ArrayView2<f32>,
    window_size: (usize, usize),
    overlap: (usize, usize),
    exponent: f32,
    normalize_power: bool,
) -> Array2<f32> {
    assert!(
        window_size.0 > 4 && window_size.1 > 4,
        "Bad window_size: {:?}. window_size has to be base 2 and > 4.",
        window_size
    );
    assert!(
        overlap.0 < (window_size.0 / 2),
        "overlap {} is too large. Maximum overlap: {}",
        overlap.0,
        window_size.0 / 2 - 1
    );
    assert!(
        overlap.1 < (window_size.1 / 2),
        "overlap {} is too large. Maximum overlap: {}",
        overlap.1,
        window_size.1 / 2 - 1
    );

    let window_stride = (window_size.0 - overlap.0, window_size.1 - overlap.1);
    let window_non_overlap = (window_size.0 - 2 * overlap.0, window_size.1 - 2 * overlap.1);
    let window_px = window_size.0 * window_size.1;
    let mut data_padded = Array2::<f32>::zeros((
        data.nrows() + 2 * window_stride.0,
        data.ncols() + 2 * window_stride.1,
    ));
    data_padded
        .slice_mut(s![
            window_stride.0..data.nrows() + window_stride.0,
            window_stride.1..data.ncols() + window_stride.1
        ])
        .assign(&data);
    let npx_x = data_padded.nrows();
    let npx_y = data_padded.ncols();

    let nx = npx_x / window_stride.0;
    let ny = npx_y / window_stride.1;

    let window_shape = (window_size.0, window_size.1);
    let taper = triangular_taper_2d(window_size, window_non_overlap);

    let mut frames = HashMap::with_capacity(nx * ny);
    for ix in 0..nx {
        let px_x_beg = cmp::min(ix * window_stride.0, npx_x);
        let px_x_end = cmp::min(px_x_beg + window_size.0, npx_x);
        for iy in 0..ny {
            let px_y_beg = cmp::min(iy * window_stride.1, npx_y);
            let px_y_end = cmp::min(px_y_beg + window_size.1, npx_y);

            let tile = Tile {
                px_x: (px_x_beg, px_x_end),
                px_y: (px_y_beg, px_y_end),
                ix: ix,
                iy: iy,
            };

            let window_data = data_padded
                .slice(s![px_x_beg..px_x_end, px_y_beg..px_y_end])
                .to_owned();
            frames.insert(tile, window_data);
        }
    }

    let fft2_r2c = R2CPlan32::aligned(&[window_size.0, window_size.1], Flag::MEASURE).unwrap();
    let fft2_c2r = C2RPlan32::aligned(&[window_size.0, window_size.1], Flag::MEASURE).unwrap();
    frames.par_iter_mut().for_each(|(_tile, window_data)| {
        let frame_shape = [window_data.shape()[0], window_data.shape()[1]];
        let fft_size = window_size.0 * (window_size.1 / 2 + 1);
        let mut window_data_fft = AlignedVec::new(fft_size);
        let mut power_spec = Array1::<f32>::default(fft_size);

        if frame_shape != [window_size.0, window_size.1] {
            let mut window_padded = Array2::<f32>::zeros(window_shape);
            window_padded
                .slice_mut(s![..frame_shape[0], ..frame_shape[1]])
                .assign(window_data);
            *window_data = window_padded;
        }

        let mut window_data_slice = window_data.as_slice_mut().unwrap();
        fft2_r2c
            .r2c(&mut window_data_slice, &mut window_data_fft)
            .unwrap();

        let mut power_max: f32 = 0.;
        for (ipx, px) in window_data_fft.iter().enumerate() {
            power_spec[ipx] = px.norm();
            power_max = power_max.max(power_spec[ipx]);
        }

        if normalize_power {
            power_spec /= power_max;
        }

        // Filter the spectrum
        for (ipx, px) in window_data_fft.iter_mut().enumerate() {
            *px *= power_spec[ipx].powf(exponent);
        }

        fft2_c2r
            .c2r(&mut window_data_fft, &mut window_data_slice)
            .unwrap();

        // Normalize fft
        *window_data /= window_px as f32;
        *window_data *= &taper;

        if frame_shape != [window_size.0, window_size.1] {
            *window_data = window_data
                .slice(s![..frame_shape[0], ..frame_shape[1]])
                .to_owned();
        }
    });

    let mut filtered_data = Array2::<f32>::zeros(data_padded.raw_dim());
    for (tile, window_data) in frames.iter() {
        let mut filtered_slice =
            filtered_data.slice_mut(s![tile.px_x.0..tile.px_x.1, tile.px_y.0..tile.px_y.1]);
        filtered_slice += window_data;
    }
    filtered_data
        .slice(s![
            window_stride.0..data.nrows() + window_stride.0,
            window_stride.1..data.ncols() + window_stride.1
        ])
        .to_owned()
}

fn afk_filter(
    data: ArrayView2<f32>,
    window_size: usize,
    overlap: usize,
    exponent: f32,
    normalize_power: bool,
) -> Array2<f32> {
    assert!(
        ((window_size as f64).log2() % 1.0 == 0.) && window_size > 4,
        "window_size has to be pow(2) and > 4."
    );
    assert!(
        overlap < (window_size / 2),
        "overlap {} is too large. Maximum overlap: {}",
        overlap,
        window_size / 2 - 1
    );

    let window_stride = window_size - overlap;
    let window_non_overlap = window_size - 2 * overlap;
    let window_px = window_size * window_size;
    let mut data_padded = Array2::<f32>::zeros((
        data.nrows() + 2 * window_stride,
        data.ncols() + 2 * window_stride,
    ));
    data_padded
        .slice_mut(s![
            window_stride..data.nrows() + window_stride,
            window_stride..data.ncols() + window_stride
        ])
        .assign(&data);
    let npx_x = data_padded.nrows();
    let npx_y = data_padded.ncols();

    let nx = npx_x / window_stride;
    let ny = npx_y / window_stride;

    let window_shape = (window_size, window_size);
    let taper = triangular_taper(window_size, window_non_overlap);

    let mut frames = HashMap::with_capacity(nx * ny);
    for ix in 0..nx {
        let px_x_beg = cmp::min(ix * window_stride, npx_x);
        let px_x_end = cmp::min(px_x_beg + window_size, npx_x);
        for iy in 0..ny {
            let px_y_beg = cmp::min(iy * window_stride, npx_y);
            let px_y_end = cmp::min(px_y_beg + window_size, npx_y);

            let tile = Tile {
                px_x: (px_x_beg, px_x_end),
                px_y: (px_y_beg, px_y_end),
                ix: ix,
                iy: iy,
            };

            let window_data = data_padded
                .slice(s![px_x_beg..px_x_end, px_y_beg..px_y_end])
                .to_owned();
            frames.insert(tile, window_data);
        }
    }

    let fft2_r2c = R2CPlan32::aligned(&[window_size, window_size], Flag::MEASURE).unwrap();
    let fft2_c2r = C2RPlan32::aligned(&[window_size, window_size], Flag::MEASURE).unwrap();
    frames.par_iter_mut().for_each(|(_tile, window_data)| {
        let frame_shape = [window_data.shape()[0], window_data.shape()[1]];
        let fft_size = (window_size / 2 + 1) * window_size;
        let mut window_data_fft = AlignedVec::new(fft_size);
        let mut power_spec = Array1::<f32>::default(fft_size);

        if frame_shape != [window_size, window_size] {
            let mut window_padded = Array2::<f32>::zeros(window_shape);
            window_padded
                .slice_mut(s![..frame_shape[0], ..frame_shape[1]])
                .assign(window_data);
            *window_data = window_padded;
        }

        let mut window_data_slice = window_data.as_slice_mut().unwrap();
        fft2_r2c
            .r2c(&mut window_data_slice, &mut window_data_fft)
            .unwrap();

        let mut power_max: f32 = 0.;
        for (ipx, px) in window_data_fft.iter().enumerate() {
            power_spec[ipx] = px.norm();
            power_max = power_max.max(power_spec[ipx]);
        }

        if normalize_power {
            power_spec /= power_max;
        }

        // Filter the spectrum and normalize
        for (ipx, px) in window_data_fft.iter_mut().enumerate() {
            *px *= power_spec[ipx].powf(exponent);
            *px /= window_px as f32;
        }

        fft2_c2r
            .c2r(&mut window_data_fft, &mut window_data_slice)
            .unwrap();

        *window_data *= &taper;

        if frame_shape != [window_size, window_size] {
            *window_data = window_data
                .slice(s![..frame_shape[0], ..frame_shape[1]])
                .to_owned();
        }
    });

    let mut filtered_data = Array2::<f32>::zeros(data_padded.raw_dim());
    for (tile, window_data) in frames.iter() {
        let mut filtered_slice =
            filtered_data.slice_mut(s![tile.px_x.0..tile.px_x.1, tile.px_y.0..tile.px_y.1]);
        filtered_slice += window_data;
    }
    filtered_data
        .slice(s![
            window_stride..data.nrows() + window_stride,
            window_stride..data.ncols() + window_stride
        ])
        .to_owned()
}

/// A Python module implemented in Rust.
#[pymodule]
fn lightguide(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "triangular_taper")]
    fn triangular_taper<'py>(
        py: Python<'py>,
        window_size: (usize, usize),
        plateau: (usize, usize),
    ) -> &'py PyArray2<f32> {
        let taper = triangular_taper_2d(window_size, plateau).into_pyarray(py);
        taper
    }

    #[pyfn(m)]
    #[pyo3(name = "afk_filter")]
    fn afk_filter_wrapper<'py>(
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
        window_size: usize,
        overlap: usize,
        exponent: f32,
        normalize_power: bool,
    ) -> &'py PyArray2<f32> {
        let data_array = data.as_array();
        let result = afk_filter(data_array, window_size, overlap, exponent, normalize_power)
            .into_pyarray(py);
        result
    }

    #[pyfn(m)]
    #[pyo3(name = "afk_filter_rectangular")]
    fn afk_filter_rectangular_wrapper<'py>(
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
        window_size: (usize, usize),
        overlap: (usize, usize),
        exponent: f32,
        normalize_power: bool,
    ) -> &'py PyArray2<f32> {
        let data_array = data.as_array();
        let result =
            afk_filter_rectangular(data_array, window_size, overlap, exponent, normalize_power)
                .into_pyarray(py);
        result
    }
    Ok(())
}
