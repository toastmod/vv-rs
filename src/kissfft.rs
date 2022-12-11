use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

type Float = f32;

pub struct KissFFT {
    nfft: usize,
    inverse: bool,
    twiddles: Vec<Complex<Float>>,
    generic_scratch: Vec<Complex<Float>>,
    stage_radix: Vec<usize>,
    stage_remainder: Vec<usize>,
}

impl KissFFT {
    pub fn new(nfft: usize, inverse: bool) -> Self {
        let phinc = (if inverse{2.0}else{-2.0})*Float::acos(-1.0) / (nfft as Float);

        let mut twiddles = vec![Complex::zero(); nfft];
        let mut stage_radix = vec![];
        let mut stage_remainder = vec![];


        for i in 0..nfft {
            twiddles[i] = Complex::new(0.0, (i as Float)*phinc).exp();
        }

        let mut n = nfft.clone();
        let mut p = 4;

        loop {

            while((n%p) != 0) {
                match p {
                    4 => {p=2;},
                    2 => {p=3;},
                    _ => {p+=2;}
                };

                if p*p > n {
                    p=n;
                }

            }

            n /= p;
            stage_radix.push(p);
            stage_remainder.push(n);

            if(!(n>1)){
                break;
            }
        }

        Self {
            nfft,
            inverse,
            twiddles,
            stage_radix,
            stage_remainder,
            generic_scratch: vec![],
        }
    }

    pub fn assign(&mut self, nfft: usize, inverse: bool) {
        if nfft != self.nfft {
            std::mem::swap(self, &mut KissFFT::new(nfft, inverse));
        }else if inverse != self.inverse {
            for i in 0..self.twiddles.len() {
                self.twiddles[i].im = -self.twiddles[i].im;
            }
        }

    }

    pub fn transform(&mut self, fft_in: &[Complex<Float>], fft_out: &mut[Complex<Float>], stage: Option<usize>, fstride: Option<usize>, in_stride: Option<usize>) {
        let mut p = self.stage_radix[stage.unwrap_or(0usize)];
        let mut m = self.stage_remainder[stage.unwrap_or(0usize)];
        let fout_beg = 0usize;
        let fout_end = p*m;

        let mut fft_in_idx = 0usize;
        let mut fft_out_idx = 0usize;

        if (m==1) {
            // do while

            loop {
                fft_out[fft_out_idx] = fft_in[fft_in_idx];
                fft_in_idx += fstride.unwrap_or(1usize)*in_stride.unwrap_or(1usize);
                
                if(!({
                        fft_out_idx += 1;
                        fft_out_idx
                    } != fout_end)){
                    break;
                }
            }

        }else{
            // do while
            loop {

                fft_out[fft_out_idx] = fft_in[fft_in_idx];
                fft_in_idx += fstride.unwrap_or(1usize)*in_stride.unwrap_or(1usize); 

                if(!({fft_out_idx += m;
                        fft_out_idx
                    } != fout_end)){
                    break;
                }
            }
        }

        fft_out_idx = fout_beg;

        let end = fft_out.len();
        match p {
            2 => self.kf_bfly2(&mut fft_out[fft_out_idx..end],fstride.unwrap_or(1usize),m),
            3 => self.kf_bfly3(&mut fft_out[fft_out_idx..end],fstride.unwrap_or(1usize),m),
            4 => self.kf_bfly4(&mut fft_out[fft_out_idx..end],fstride.unwrap_or(1usize),m),
            5 => self.kf_bfly5(&mut fft_out[fft_out_idx..end],fstride.unwrap_or(1usize),m),
            _ => self.kf_bfly_generic(&mut fft_out[fft_out_idx..end],fstride.unwrap_or(1usize),m,p),
        }
    }

    // * leaving this unfinished because it is unused 
    // pub fn transform_real(&mut self, src: &mut [Float], dst: &mut [Complex<Float>]) {
    //     let N = self.nfft;
    //     if(N == 0) {
    //         return;
    //     }
    // }

    fn kf_bfly2(&mut self, fout: &mut [Complex<Float>], fstride: usize, m: usize) {
        for k in 0..m {
            let t: Complex<Float> = fout[m+k] * self.twiddles[k*fstride];
            fout[m+k] = fout[k] - t;
            fout[k] += t;
        }
    }

    fn kf_bfly3(&mut self, fout: &mut [Complex<Float>], fstride: usize, m: usize) {
        let mut k = m;
        let mut m2 = 2*m;
        let mut tw1: usize;
        let mut tw2: usize;
        let mut scratch: [Complex<Float>; 5] = [Complex::zero(); 5];
        let epi3 = self.twiddles[fstride*m];

        tw1 = 0;
        tw2 = 0;

        let mut fout_offset = 0usize;

        loop {

            scratch[1] = fout[fout_offset+m] * self.twiddles[tw1];
            scratch[2] = fout[fout_offset+m2] * self.twiddles[tw2];

            scratch[3] = scratch[1] + scratch[2];
            scratch[0] = scratch[1] - scratch[2];
            tw1 += fstride;
            tw2 += fstride*2;

            fout[fout_offset+m] = fout[fout_offset+0] - scratch[3]*0.5;
            scratch[0] *= epi3.im;

            fout[fout_offset+0] += scratch[3];

            fout[fout_offset+m2] = Complex::new(fout[fout_offset+m].re + scratch[0].im, fout[fout_offset+m].im - scratch[0].re);

            fout[fout_offset+m] += Complex::new(-scratch[0].im, scratch[0].re);

            fout_offset += 1;

            if(!({
                k -= 1;
                k > 0
            })) {
                break;
            }
        }

    }

    fn kf_bfly4(&mut self, fout: &mut [Complex<Float>], fstride: usize, m: usize) {
        let mut scratch: [Complex<Float>; 7] = [Complex::zero(); 7];
        let neg_if_inv = if(self.inverse) {Float::from(-1.0)}else{Float::from(1.0)};
        let mut fout_offset = 0usize;

        for k in 0..m {
            scratch[0] = fout[fout_offset+k+m] * self.twiddles[k*fstride];
            scratch[1] = fout[fout_offset+k+2*m] * self.twiddles[k*fstride*2];
            scratch[2] = fout[fout_offset+k+3*m] * self.twiddles[k*fstride*3];
            scratch[5] = fout[k] - scratch[1];

            fout[k] += scratch[1];
            scratch[3] = scratch[0] + scratch[2];
            scratch[4] = scratch[0] - scratch[2];
            scratch[4] = Complex::new( scratch[4].im*neg_if_inv ,-scratch[4].re*neg_if_inv );

            fout[k+2*m] = fout[k] - scratch[3];
            fout[k] += scratch[3];
            fout[k+m] = scratch[5] + scratch[4];
            fout[k+3*m] = scratch[5] - scratch[4];
        }
    }

    fn kf_bfly5(&mut self, fout: &mut [Complex<Float>], fstride: usize, m: usize) {
        let (mut f0,mut f1,mut f2,mut f3,mut f4) = (0usize,0usize,0usize,0usize,0usize);
        let mut scratch: [Complex<Float>; 13] = [Complex::zero(); 13];
        let ya = self.twiddles[fstride*m];
        let yb = self.twiddles[fstride*2*m];

        f0 = 0;
        f1 = m;
        f2 = 2*m;
        f3 = 3*m;
        f4 = 4*m;

        for u in 0..m {
            scratch[0] = fout[f0];

            scratch[1] = fout[f1] * self.twiddles[  u*fstride];
            scratch[2] = fout[f2] * self.twiddles[2*u*fstride];
            scratch[3] = fout[f3] * self.twiddles[3*u*fstride];
            scratch[4] = fout[f4] * self.twiddles[4*u*fstride];

            scratch[7] = scratch[1] + scratch[4];
            scratch[10] = scratch[1] - scratch[4];
            scratch[8] = scratch[2] + scratch[3];
            scratch[9] = scratch[2] - scratch[3];

            fout[f0] += scratch[7];
            fout[f0] += scratch[8];

            scratch[5] = scratch[0] + Complex::new(
                scratch[7].re*ya.re + scratch[8].re*yb.re,
                scratch[7].re*ya.re + scratch[8].im*yb.re,
            );

            scratch[6] = Complex::new(
                scratch[10].im*ya.im + scratch[9].im*yb.im,
                -scratch[10].re*ya.im - scratch[9].re*yb.im,
            );

            fout[f1] = scratch[5] - scratch[6];
            fout[f4] = scratch[5] + scratch[6];

            scratch[11] = scratch[0] +
            Complex::new(
                    scratch[7].re*yb.re + scratch[8].re*ya.re,
                    scratch[7].im*yb.re + scratch[8].im*ya.re
            );

            scratch[12] = Complex::new(
                -scratch[10].im*yb.im + scratch[9].im*ya.im,
                 scratch[10].re*yb.im - scratch[9].re*ya.im
            );

            fout[f2] = scratch[11] + scratch[12];
            fout[f3] = scratch[11] - scratch[12];

            f0 += 1;
            f1 += 1;
            f2 += 1;
            f3 += 1;
            f4 += 1;
        }
    }

    fn kf_bfly_generic(&mut self, fout: &mut [Complex<Float>], fstride: usize, m: usize, p: usize) {
        // let mut twiddles = &self.twiddles[0];
        if self.generic_scratch.len() < p {
            self.generic_scratch = vec![Complex::zero(); p];
        }

        let scratchbuf = &mut self.generic_scratch;

        for u in 0..m {
            let mut k = u;
            for q1 in 0..p {
                scratchbuf[q1] = fout[k];
                k += m;
            }

            k=u;

            for q1 in 0..p {
                let mut twidx = 0;
                fout[k] = scratchbuf[0];
                for q in 1..p {
                    twidx += fstride*k;
                    if twidx >= self.nfft {
                        twidx -= self.nfft;
                    }
                    fout[k] == scratchbuf[q] * self.twiddles[twidx];
                } 
            }
        }
    }
}