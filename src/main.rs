use std::f32::consts::PI;

use bevy::prelude::*;
use cpal::Sample;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender, unbounded};
use rustfft::{FftPlanner, num_complex::Complex};

const BAR_COUNT: usize = 48;
const FFT_SIZE: usize = 1024;
const SAMPLE_BUFFER_LIMIT: usize = FFT_SIZE * 8;
const WINDOW_WIDTH: f32 = 1280.0;
const WINDOW_HEIGHT: f32 = 800.0;
const BAR_WIDTH: f32 = 18.0;
const BAR_GAP: f32 = 6.0;
const BOTTOM_MARGIN: f32 = 34.0;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.03, 0.04, 0.07)))
        .insert_resource(AudioSpectrum {
            receiver: None,
            samples: Vec::new(),
            spectrum: Vec::new(),
            stream_status: String::new(),
        })
        .insert_resource(FftProcessor::new())
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Mic Frequency Visualizer".into(),
                resolution: (WINDOW_WIDTH as u32, WINDOW_HEIGHT as u32).into(),
                resizable: true,
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, setup)
        .add_systems(Update, (drain_audio, update_bars))
        .add_systems(PostUpdate, check_audio_status)
        .run();
}

#[derive(Component)]
struct FrequencyBar {
    index: usize,
    x: f32,
}

#[derive(Resource)]
struct AudioSpectrum {
    receiver: Option<Receiver<Vec<f32>>>,
    samples: Vec<f32>,
    spectrum: Vec<f32>,
    stream_status: String,
}

#[derive(Resource)]
struct FftProcessor {
    planner: FftPlanner<f32>,
    window: Vec<f32>,
    buffer: Vec<Complex<f32>>,
}

impl FftProcessor {
    fn new() -> Self {
        // Precompute Hann window
        let mut window = vec![0.0; FFT_SIZE];
        for i in 0..FFT_SIZE {
            window[i] = 0.5 - 0.5 * (2.0 * PI * i as f32 / (FFT_SIZE as f32 - 1.0)).cos();
        }

        Self {
            planner: FftPlanner::new(),
            window,
            buffer: vec![Complex::new(0.0, 0.0); FFT_SIZE],
        }
    }
}

// Resource to hold the audio stream (keeps it alive)
// cpal::Stream is intentionally not Send + Sync, but we need it to be a Resource.
// Safety: The stream is only accessed by cpal's internal audio callbacks running on
// a dedicated thread pool. Bevy never moves or accesses it directly; it only holds the
// reference to keep it alive. The stream is safely dropped when this resource is dropped.
#[derive(Resource)]
struct AudioStream {
    _stream: Option<cpal::Stream>,
}

// Safety: See comment above. The stream manages its own thread safety internally.
unsafe impl Send for AudioStream {}
unsafe impl Sync for AudioStream {}

fn setup(mut commands: Commands, mut spectrum: ResMut<AudioSpectrum>) {
    commands.spawn(Camera2d);

    let bar_span = BAR_WIDTH + BAR_GAP;
    let total_width = BAR_COUNT as f32 * bar_span - BAR_GAP;
    let start_x = -total_width / 2.0 + BAR_WIDTH / 2.0;

    for index in 0..BAR_COUNT {
        let x = start_x + index as f32 * bar_span;
        commands.spawn((
            Sprite::from_color(Color::srgb(0.22, 0.72, 0.95), Vec2::new(BAR_WIDTH, 2.0)),
            Transform::from_xyz(x, -WINDOW_HEIGHT / 2.0 + BOTTOM_MARGIN, 0.0),
            FrequencyBar { index, x },
        ));
    }

    let (sender, receiver) = unbounded::<Vec<f32>>();
    spectrum.receiver = Some(receiver);
    spectrum.stream_status = "initializing audio...".into();

    match initialize_audio_stream(sender) {
        Ok(stream) => {
            commands.insert_resource(AudioStream {
                _stream: Some(stream),
            });
        }
        Err(e) => {
            spectrum.stream_status = format!("error: {}", e);
            eprintln!("Failed to initialize audio: {}", e);
        }
    }
}

fn initialize_audio_stream(sender: Sender<Vec<f32>>) -> Result<cpal::Stream, String> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("No input device available")?;

    let config = device
        .default_input_config()
        .map_err(|e| format!("Failed to read input config: {}", e))?;

    let channels = config.channels() as usize;
    let stream_config: cpal::StreamConfig = config.clone().into();

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => {
            build_stream::<f32>(&device, &stream_config, channels, sender)
        }
        cpal::SampleFormat::I16 => {
            build_stream::<i16>(&device, &stream_config, channels, sender)
        }
        cpal::SampleFormat::U16 => {
            build_stream::<u16>(&device, &stream_config, channels, sender)
        }
        sample_format => {
            return Err(format!("Unsupported sample format: {:?}", sample_format));
        }
    }
    .map_err(|e| format!("Failed to build stream: {}", e))?;

    stream
        .play()
        .map_err(|e| format!("Failed to start stream: {}", e))?;

    Ok(stream)
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    sender: Sender<Vec<f32>>,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: Sample + cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    device.build_input_stream(
        config,
        move |data: &[T], _| {
            let mut mono = Vec::with_capacity(data.len() / channels.max(1));

            for frame in data.chunks(channels.max(1)) {
                let mut sum = 0.0;
                for sample in frame {
                    sum += sample.to_sample::<f32>();
                }
                mono.push(sum / frame.len() as f32);
            }

            let _ = sender.try_send(mono);
        },
        move |error| {
            eprintln!("Audio input error: {error}");
        },
        None,
    )
}

fn drain_audio(mut spectrum: ResMut<AudioSpectrum>, mut fft: ResMut<FftProcessor>) {
    let Some(receiver) = spectrum.receiver.as_ref().cloned() else {
        return;
    };

    while let Ok(chunk) = receiver.try_recv() {
        spectrum.samples.extend(chunk);
    }

    if spectrum.samples.len() > SAMPLE_BUFFER_LIMIT {
        let excess = spectrum.samples.len() - SAMPLE_BUFFER_LIMIT;
        spectrum.samples.drain(..excess);
    }

    if spectrum.samples.len() < FFT_SIZE {
        return;
    }

    spectrum.spectrum = analyze_spectrum(
        &spectrum.samples[spectrum.samples.len() - FFT_SIZE..],
        &mut fft,
    );
    spectrum.stream_status = "mic live".into();
}

fn analyze_spectrum(samples: &[f32], fft_processor: &mut FftProcessor) -> Vec<f32> {
    let fft = fft_processor.planner.plan_fft_forward(FFT_SIZE);

    // Reuse buffer and apply windowing
    for (i, sample) in samples.iter().enumerate() {
        fft_processor.buffer[i] = Complex::new(sample * fft_processor.window[i], 0.0);
    }

    fft.process(&mut fft_processor.buffer);

    let half = FFT_SIZE / 2;
    let bins_per_bar = half as f32 / BAR_COUNT as f32;
    let mut bars = vec![0.0; BAR_COUNT];

    for (index, bar) in bars.iter_mut().enumerate() {
        let start = (index as f32 * bins_per_bar).floor() as usize;
        let end = ((index as f32 + 1.0) * bins_per_bar).ceil() as usize;
        let clamped_end = end.clamp(start + 1, half);

        let mut total = 0.0;
        let mut count = 0;

        for bin in start..clamped_end {
            total += fft_processor.buffer[bin].norm();
            count += 1;
        }

        let average = if count == 0 {
            0.0
        } else {
            total / count as f32
        };
        *bar = (average * 0.03).clamp(0.0, 1.0);
    }

    bars
}

fn update_bars(
    spectrum: Res<AudioSpectrum>,
    mut bars: Query<(&FrequencyBar, &mut Sprite, &mut Transform)>,
) {
    for (bar, mut sprite, mut transform) in &mut bars {
        let level = spectrum.spectrum.get(bar.index).copied().unwrap_or(0.0);
        let height = 12.0 + level.powf(0.65) * 640.0;

        sprite.custom_size = Some(Vec2::new(BAR_WIDTH, height));
        sprite.color = Color::srgb(0.12 + level * 0.85, 0.45 + level * 0.4, 0.85 - level * 0.45);
        transform.translation.x = bar.x;
        transform.translation.y = -WINDOW_HEIGHT / 2.0 + BOTTOM_MARGIN + height / 2.0;
    }
}

fn check_audio_status(audio_stream: Option<Res<AudioStream>>, mut spectrum: ResMut<AudioSpectrum>) {
    if audio_stream.is_some() && spectrum.stream_status == "initializing audio..." {
        spectrum.stream_status = "mic live".into();
    }
}
