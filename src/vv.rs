use std::arch::x86_64::_bextr_u32;
use std::sync::Arc;
use rustfft;
use rustfft::Fft;
use rustfft::num_complex::{Complex, ComplexFloat};
use rustfft::num_traits::{FloatConst, Zero};

pub struct FormantProc {
    dummy: Vec<Complex<f32>>,
    sample_rate: usize,
    pub output: Vec<f32>,
    v1: Vec<Complex<f32>>,
    v2: Vec<Complex<f32>>,
    v3: Vec<Complex<f32>>,
    // scratchv: Vec<Complex<f32>>,
    // iscratchv: Vec<Complex<f32>>,
    v4: Vec<Complex<f32>>,
    v5: Vec<Complex<f32>>,
    v6: Vec<f32>,
    v7: Vec<f32>,
    buffer_size: usize,
    nsdf_size: usize,
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
    last_peak_index: Option<usize>,
}

impl FormantProc {
    pub fn new(buffer_size: usize, sample_rate: usize) -> Self {

        let nsdf_size = buffer_size/2;
        let fft = rustfft::FftPlanner::new().plan_fft_forward(buffer_size + nsdf_size);
        let ifft = rustfft::FftPlanner::new().plan_fft_inverse(buffer_size + nsdf_size);
        // let scratch_size = fft.get_outofplace_scratch_len();
        // let iscratch_size = ifft.get_outofplace_scratch_len();
        Self {
            dummy: vec![Complex::zero(); buffer_size + nsdf_size],
            sample_rate,
            output: vec![0f32; buffer_size],
            v1: vec![Complex::zero(); buffer_size],
            v2: vec![Complex::zero(); buffer_size + nsdf_size],
            v3: vec![Complex::zero(); buffer_size + nsdf_size],
            // scratchv: vec![Complex::zero(); buffer_size + nsdf_size],
            // iscratchv: vec![Complex::zero(); buffer_size + nsdf_size],
            v4: vec![Complex::zero(); buffer_size + nsdf_size],
            v5: vec![Complex::zero(); buffer_size + nsdf_size],
            v6: vec![0f32; buffer_size],
            v7: vec![0f32; nsdf_size],
            buffer_size,
            nsdf_size,
            fft,
            ifft,
            last_peak_index: None
        }
    }

    fn norm(v: Complex<f32>) -> Complex<f32> {
        // Complex::new((v.re * v.re) + (v.im * v.im), 0.0)
        Complex::new(v.abs() * v.abs(), 0.0)
    }

    fn easing(x: f64) -> f64 {
        (1.0 - f64::cos(f64::PI() * x)) / 2f64
    }

    fn lerp(a: f64, b: f64, f: f64) -> f64 {
        a + f * (b - a)
    }

    fn interpolate(x1: f64, x2: f64, ratio: f64) -> f64 {
        Self::lerp(x1, x2, Self::easing(ratio))
    }

    fn get_value(&self, indexf: f64, buffer: &mut [f32]) -> f64 {
        if(indexf < 0.0) {
            return buffer[0] as f64;
        }

        let index = f64::floor(indexf) as usize;

        if index >= (self.buffer_size - 1) {
            return buffer[self.buffer_size - 1] as f64;
        }

        let ratio = indexf - f64::floor(indexf);

        Self::interpolate(buffer[index] as f64, buffer[index+1] as f64, ratio)

    }

    pub fn process(&mut self, buffer: &mut [f32], formant_shift: f64, pitch_shift: f64) {

        for i in 0..self.buffer_size {
            self.v1[i] = Complex::new(buffer[i], 0.0);
        }

        // ! KissFFT uses 2*PI as it's maximum, RustFFT does not use this.
        for i in 0..self.buffer_size {
            let r = (i as f64) / (self.buffer_size as f64);
            let w = 0.5-(0.5*f64::cos(2f64*std::f64::consts::PI * r));
            self.v2[i] = self.v1[i]* (w as f32);
            assert_eq!(false,self.v2[i].re.is_nan());
        }

        self.v3[0..self.buffer_size].copy_from_slice(self.v1.as_slice());
        self.v3[self.buffer_size..self.buffer_size+self.nsdf_size].copy_from_slice(&self.dummy.as_slice()[0..self.nsdf_size]);
        self.fft.process(self.v3.as_mut_slice());

        let cutoff_hz = 800f64;
        let cutoff_index = f64::round(cutoff_hz * self.buffer_size as f64 / self.sample_rate as f64) as usize;

        for i in 0..cutoff_index {
            self.v4[i+1] = Self::norm(self.v3[i+1]);
            self.v4[self.buffer_size - i -1] = Self::norm(self.v3[self.buffer_size - i - 1]);
        }

        self.v5.copy_from_slice(self.v4.as_slice());
        self.ifft.process(self.v5.as_mut_slice());
        for i in 0..(self.buffer_size + self.nsdf_size) {
            self.v5[i] /= ((self.buffer_size+self.nsdf_size) as f32);
        }

        for i in 1..self.buffer_size {
            let j = self.buffer_size - i - 1;
            let n1 = (self.v2[i].re);
            let n2 = (self.v2[j].re);
            self.v6[j] = self.v6[j+1] + (n1*n1) + (n2*n2);
        }

        for i in 0..(self.buffer_size/2) {
            if self.v6[i] < f32::MIN {
                self.v7[i] = 0.0;
            }else{
                self.v7[i] = 2f32 * self.v5[i].re/self.v6[i];
            }
        }

        let min_hz = 50.0;
        let max_hz = 300.0;

        let mut min_index = f64::round(self.sample_rate as f64 / max_hz) as usize;
        let mut max_index = f64::round(self.sample_rate as f64 / min_hz) as usize;

        min_index = usize::max(min_index, 1);
        min_index = usize::min(min_index, (self.buffer_size/2) - 2);

        max_index = usize::max(max_index, 1);
        max_index = usize::min(max_index, (self.buffer_size/2) - 2);

        let mut max_val = 0f64;

        for i in min_index..max_index {
            let p1 = self.v7[i-1] as f64;
            let p2 = self.v7[i] as f64;
            let p3 = self.v7[i+1] as f64;

            if (p1<p2) && (p2>p3) && (p2 > max_val) {
                max_val = p2;
            }
        }

        let mut peak_index: Option<usize> = None;
        let mut peak_value = 0f64;

        for i in min_index..max_index {
            let p1 = self.v7[i - 1] as f64;
            let p2 = self.v7[i] as f64;
            let p3 = self.v7[i + 1] as f64;

            if (p1<p2) && (p2>p3) && (p2 > (max_val * 0.9)) {
                peak_index = Some(i);
                peak_value = p2;
                break;
            }
        }

        if peak_index.is_none() && self.last_peak_index.is_some() {
            peak_index = self.last_peak_index.clone();
        }

        self.last_peak_index = peak_index.clone();

        let mut enable = false;

        if(peak_index.is_some()) {
            let mut last_dst = 0;
            let mut last_src1 = 0;
            let mut last_src2 = 0;
            let mut last_src_ratio = 0f64;

            let bufsize = self.buffer_size.clone();

            let mut overlap = |dst: usize, src1: usize, src2: usize, src_ratio: f64| {
                for i in last_dst..dst {
                    let ratio = (i-last_dst) as f64 / (dst-last_dst) as f64;

                    let p1_1 = self.get_value(last_src1 as f64 + (i-last_dst) as f64 * formant_shift, buffer);
                    let p1_2 = self.get_value(last_src2 as f64 + (i-last_dst) as f64 * formant_shift, buffer);
                    let p1 = Self::interpolate(p1_1, p1_2, last_src_ratio);

                    let p2_1 = self.get_value(src1 as f64 + (dst - i) as f64 * formant_shift, buffer);
                    let p2_2 = self.get_value(src2 as f64 + (dst - i) as f64 * formant_shift, buffer);
                    let p2 = Self::interpolate(p2_1, p2_2, src_ratio);

                    self.output[i] = (Self::interpolate(p1,p2,ratio) as f32);

                }

                last_dst = dst;
                last_src1 = src1;
                last_src2 = src2;
                last_src_ratio = src_ratio;

            };

            let peak_index_now = peak_index.unwrap_or(1);
            let q = bufsize / peak_index_now;
            let r = bufsize % peak_index_now;

            let nf = bufsize as f64 * pitch_shift - r as f64 / peak_index_now as f64;
            let n = f64::max(0.0, f64::round(nf)) as usize;

            if (q != 0) && (n != 0) {
                let actual_pitch_shift = (n * peak_index_now + r) as f64 / bufsize as f64;

                for i in 1..=n {
                    let mut frame_indexf = 0.0;
                    if(n != 1) {
                        frame_indexf = ((i-1)*(q-1)) as f64 / ((n-1)+1) as f64;
                    }

                    let frame_index = f64::floor(frame_indexf) as usize;

                    let dst = (f64::floor(i as f64 * peak_index_now as f64) / actual_pitch_shift) as usize;
                    let src = frame_index * peak_index_now;

                    if(frame_index == q) {
                        overlap(dst, src, src, 0.0);
                    }else{
                        let src_ratio = frame_indexf - f64::floor(frame_indexf);
                        overlap(dst, src, src+peak_index_now, src_ratio);
                    }
                }

                overlap(bufsize, bufsize, bufsize, 0.0);

                enable = true;

            }


        }

        if(!enable) {
            self.output.copy_from_slice(buffer);
        }

    }

}