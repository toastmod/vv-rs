use crate::util::do_while;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

type Float = f32;

pub struct KissFFT {
    nfft: usize,
    inverse: bool,
    twiddles: Vec<Complex<Float>>,
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

        do_while(
            &mut || {
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
            },
            &mut || n>1
        );

        Self {
            nfft,
            inverse,
            twiddles,
            stage_radix,
            stage_remainder
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

            do_while(
                &mut ||{
                    fft_out[fft_out_idx] = fft_in[fft_in_idx];
                    fft_in_idx += fstride.unwrap_or(1usize)*in_stride.unwrap_or(1usize);
                },
                &mut ||{
                    ({
                        fft_out_idx += 1;
                        fft_out_idx
                    } != fout_end)
                }
            );
        }else{
            // do while
            do_while(
                &mut ||{
                    fft_out[fft_out_idx] = fft_in[fft_in_idx];
                    fft_in_idx += fstride.unwrap_or(1usize)*in_stride.unwrap_or(1usize);
                },

                &mut || {
                    ({
                        fft_out_idx += m;
                        fft_out_idx
                    } != fout_end)
                }
            );
        }

        fft_out_idx = fout_beg;

        match p {
            2 => self.kf_bfly2(fft_out,fft_out_idx,fstride,m),
            3 => self.kf_bfly3(fft_out,fft_out_idx,fstride,m),
            4 => self.kf_bfly4(fft_out,fft_out_idx,fstride,m),
            5 => self.kf_bfly5(fft_out,fft_out_idx,fstride,m),
            _ => self.kf_bfly_generic(fft_out,fft_out_idx,fstride,m,p),
        }
    }

    // * leaving this unfinished because it is unused 
    // pub fn transform_real(&mut self, src: &mut [Float], dst: &mut [Complex<Float>]) {
    //     let N = self.nfft;
    //     if(N == 0) {
    //         return;
    //     }
    // }

    fn kf_bfly2(&mut self, fout: &mut [Complex<Float>], fstrid: usize, m: usize) {
        for k in 0..m {
            let t: Complex<Float> = fout[m+k] * self.twiddles[k*fstride];
            fout[m+k] = fout[k] - t;
            fout[k] += t;
        }
    }

    fn kf_bfly3(&mut self, fout: &mut [Complex<Float>], fstrid: usize, m: usize) {

    }
    

}