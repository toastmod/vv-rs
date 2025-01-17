#[cfg(test)]
mod tests {
    #[test]
    fn sin_test() {
        let bufsize = 4096;
        let mut proc = crate::vv::FormantProc::new(bufsize, 44100);
        let mut buf = vec![0f32; bufsize];

        for i in 0..4096 {
            let ratio = i as f64 / bufsize as f64;
            let v = f64::sin(10.0 * ratio * std::f64::consts::PI*2.0);

            buf[i] = v as f32;
        }

        proc.process(&mut buf, 1.5, 1.2);

        for i in 0..bufsize {
            println!("{}",buf[i]);
        }

    }
}

pub mod vv;

use nih_plug::prelude::*;
use vv::FormantProc;
use std::{sync::Arc, collections::VecDeque};

struct VV {
    params: Arc<VVParams>,
    proc: FormantProc,
    sample_rate: usize,
    buffer_size: usize,
    input_buffer: Vec<f32>,
    output_buffer: VecDeque<f32>,
    cursor: usize,
}

#[derive(Params)]
struct VVParams {

    #[id = "Formant"]
    pub formant: FloatParam,

    #[id = "FormantAdd"]
    pub fadd: FloatParam,

    #[id = "Pitch"]
    pub pitch: FloatParam,

    #[id = "PitchAdd"]
    pub padd: FloatParam,
}

impl Default for VV {
    fn default() -> Self {
        Self {
            params: Arc::new(VVParams::default()),
            proc: FormantProc::new(1024, 44100),
            sample_rate: 0,
            buffer_size: 0,
            input_buffer: vec![],
            output_buffer: VecDeque::new(),
            cursor: 0, 
        }
    }
}

impl Default for VVParams {
    fn default() -> Self {
        Self {
            formant: FloatParam::new(
                "Formant",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.5 }
            ),
            fadd: FloatParam::new(
                "FormantAdd",
                0.0,
                FloatRange::Linear { min: 0.0, max: 5.0 }
            ),
            pitch: FloatParam::new(
                "Pitch",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.5 }
            ),
            padd: FloatParam::new(
                "PitchAdd",
                0.0,
                FloatRange::Linear { min: 0.0, max: 5.0 }
            )
        }
    }
}

impl Plugin for VV {
    const NAME: &'static str = "vvrs";
    const VENDOR: &'static str = "Andrew Numrich";
    const URL: &'static str = "";
    const EMAIL: &'static str = "anumrich@hotmail.com";

    const VERSION: &'static str = "0.0.1";

    const DEFAULT_INPUT_CHANNELS: u32 = 2;
    const DEFAULT_OUTPUT_CHANNELS: u32 = 2;

    const DEFAULT_AUX_INPUTS: Option<AuxiliaryIOConfig> = None;
    const DEFAULT_AUX_OUTPUTS: Option<AuxiliaryIOConfig> = None;

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn accepts_bus_config(&self, config: &BusConfig) -> bool {
        // This works with any symmetrical IO layout
        config.num_input_channels == config.num_output_channels && config.num_input_channels > 0
    }

    fn initialize(
        &mut self,
        _bus_config: &BusConfig,
        _buffer_config: &BufferConfig,
        _context: &mut impl InitContext<VV>,
    ) -> bool {

        self.sample_rate = _buffer_config.sample_rate as usize;
        self.buffer_size = match _buffer_config.min_buffer_size {
            Some(bufsize) => bufsize as usize,
            None => _buffer_config.max_buffer_size as usize,
        }; 

        self.input_buffer = vec![0f32; self.buffer_size];

        true
    }

    fn reset(&mut self) {
        self.proc = FormantProc::new(self.buffer_size, self.sample_rate);
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<VV>,
    ) -> ProcessStatus {
        let mut chunks = buffer.as_slice();

        let formant_shift = self.params.formant.value() as f64 + self.params.fadd.value() as f64;
        let pitch_shift = self.params.pitch.value() as f64 + self.params.padd.value() as f64;

        for i in 0..chunks[0].len() {
            self.input_buffer[self.cursor] = chunks[0][i];
            self.cursor += 1;
            if self.cursor == self.buffer_size {
                self.proc.process(&mut self.input_buffer, formant_shift, pitch_shift);
                for x in 0..self.buffer_size {
                    self.output_buffer.push_back(self.input_buffer[i]);
                }
                self.cursor = 0;
            }

            let os = match self.output_buffer.pop_front() {
                Some(s) => s,
                None => 0f32,
            };

            chunks[0][i] = os;
            chunks[1][i] = os;
        }
        
        

        ProcessStatus::Normal
    }

    const PORT_NAMES: PortNames = PortNames {
        main_input: None,
        main_output: None,
        aux_inputs: None,
        aux_outputs: None,
    };

    const HARD_REALTIME_ONLY: bool = false;

    fn task_executor(&self) -> TaskExecutor<Self> {
        Box::new(|_| ())
    }

    fn editor(&self, async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        None
    }

    fn filter_state(state: &mut PluginState) {}

    fn deactivate(&mut self) {}

    type BackgroundTask = ();
}

impl ClapPlugin for VV {
    const CLAP_ID: &'static str = "com.your-domain.vv-rs";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("A fork of vv by planaria.");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;

    // Don't forget to change these features
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect, ClapFeature::Stereo];
}

impl Vst3Plugin for VV {
    const VST3_CLASS_ID: [u8; 16] = *b"vvrsplanaria!!!!";
    const VST3_CATEGORIES: &'static str = "Fx|Misc";
}

nih_export_clap!(VV);
nih_export_vst3!(VV);
