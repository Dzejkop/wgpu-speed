use std::num::NonZeroU64;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use rand::RngCore;
use wgpu::util::DeviceExt;

async fn load_data(path: impl AsRef<Path>) -> anyhow::Result<Vec<u8>> {
    let data = tokio::fs::read(path).await?;

    Ok(data)
}

/// Max number of workgroups that can be dispatched in a single compute pass
/// actual value is (1024 * 64) - 1. But to have nicer numbers will use an even value
const WORKGROUP_DISPATCH_LIMIT: u64 = 1024 * 32;

/// The compute shader workgroup size
const WORKGROUP_SIZE: u32 = 32;

/// How much data to be written to a buffer
/// 200 MiB
const DATA_SIZE: usize = 1024 * 1024 * 200;

async fn run() -> anyhow::Result<()> {
    let mut rng = rand::thread_rng();
    let mut rand_data = vec![0u8; DATA_SIZE];

    rng.fill_bytes(&mut rand_data);

    let (device, queue) = fetch_device().await?;

    let device = Arc::new(device);

    start_polling_job(device.clone());

    let now = std::time::Instant::now();
    let pipeline = Pipeline::initialize(device, queue).await?;
    let elapsed = now.elapsed();
    println!("Initialization took: {:?}", elapsed);

    let now = std::time::Instant::now();
    pipeline.run(&rand_data).await;
    println!("Computation took: {:?}", now.elapsed());

    Ok(())
}

async fn fetch_device() -> anyhow::Result<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .context("Missing adapter")?;

    println!("adapter = {:?}", adapter);
    println!("adapter.get_info() = {:#?}", adapter.get_info());

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::PUSH_CONSTANTS | wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
                limits: wgpu::Limits {
                    max_storage_buffer_binding_size: DATA_SIZE as u32,
                    max_push_constant_size: 4,
                    ..wgpu::Limits::default()
                },
            },
            None,
        )
        .await?;

    Ok((device, queue))
}

fn start_polling_job(device: Arc<wgpu::Device>) {
    tokio::task::spawn_blocking(move || loop {
        device.poll(wgpu::Maintain::Wait);
    });
}

struct Pipeline {
    buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,

    compute_pipeline: wgpu::ComputePipeline,

    bind_group: wgpu::BindGroup,

    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
}

impl Pipeline {
    pub async fn initialize(device: Arc<wgpu::Device>, queue: wgpu::Queue) -> anyhow::Result<Self> {
        let shader: wgpu::ShaderModuleDescriptor = wgpu::include_wgsl!("./shader.wgsl");

        let cs_module = device.create_shader_module(shader);

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Input buffer"),
            size: DATA_SIZE as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output buffer"),
            size: 4, // 4 Bytes for the result u32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..4,
                }],
            });

        // Instantiates the pipeline.
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&compute_pipeline_layout),
            module: &cs_module,
            entry_point: "hamming",
        });

        let staging_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[0u32; 1]),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        });

        Ok(Self {
            buffer,
            output_buffer,
            staging_buffer,

            device,
            queue,

            compute_pipeline,

            bind_group,
        })
    }

    pub async fn run<'a>(&self, input_data: &'a [u8]) {
        println!("Start");
        let now = std::time::Instant::now();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.queue.write_buffer(&self.buffer, 0, &input_data);

        self.queue.submit(None);

        println!("Time to data in buffers: {:?}", now.elapsed());

        let len = input_data.len();

        // Input data is in bytes and we're processing in u32s. So we first divide by 4 and then by workgroup size.
        let total_workgroups_to_dispatch = len / (4 * WORKGROUP_SIZE as usize);

        let chunk_data_size = WORKGROUP_SIZE * 4;
        let chunk_size = WORKGROUP_DISPATCH_LIMIT as usize;
        let num_chunks = total_workgroups_to_dispatch / chunk_size;

        println!("Time create encoder: {:?}", now.elapsed());

        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });

            cpass.set_pipeline(&self.compute_pipeline);

            cpass.set_bind_group(0, &self.bind_group, &[]);

            for i in 0..num_chunks as u32 {
                let offset = i * chunk_data_size;
                let num_workgroups = chunk_size as u32;

                self.run_inner(&mut cpass, offset, num_workgroups).await;
            }
        }

        println!("Time to finish dispatching workgroups: {:?}", now.elapsed());

        encoder.copy_buffer_to_buffer(&self.output_buffer, 0, &self.staging_buffer, 0, 4);

        self.queue.submit(Some(encoder.finish()));

        println!("Time to queue submit: {:?}", now.elapsed());

        let result_buffer = self.staging_buffer.slice(..);

        let (sender, receiver) = tokio::sync::oneshot::channel();

        result_buffer.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        receiver.await.unwrap().unwrap();
        println!("Time to read statging buffers: {:?}", now.elapsed());

        let result_mapped_range = result_buffer.get_mapped_range();

        let result_slice: &[u32] = bytemuck::cast_slice(&result_mapped_range);

        let sum_of_ones = result_slice[0];

        println!("Time to copy staging buffers: {:?}", now.elapsed());

        drop(result_mapped_range);
        self.staging_buffer.unmap();

        println!("Time to unmap statging buffers: {:?}", now.elapsed());

        println!("sum_of_ones = {}", sum_of_ones);
    }

    async fn run_inner<'a>(
        &self,
        cpass: &mut wgpu::ComputePass<'a>,
        offset: u32,
        num_workgroups: u32,
    ) {
        let pc = &[offset];
        let pc = bytemuck::cast_slice(pc);

        // TODO: Use offsets instead of push constants
        cpass.set_push_constants(0, pc);
        cpass.dispatch_workgroups(num_workgroups, 1, 1);
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    run().await?;

    Ok(())
}
