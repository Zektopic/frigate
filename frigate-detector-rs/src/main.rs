//! Threaded Rust detector — ncnn via subprocess + rayon post-process.
//! Drop-in replacement for Python YOLO worker subprocess.
//!
//! stdin/stdout pipe protocol compatible with ONNX plugin.

use crossbeam::channel::{bounded, Sender, Receiver};
use rayon::prelude::*;
use std::io::{self, Read, Write, BufReader, BufRead};
use std::process::{Command, Stdio, Child, ChildStdin, ChildStdout};
use std::sync::Arc;
use std::time::Instant;

// ── Ncnn subprocess (Python — just forward pass, no numpy post) ───

struct NcnnProc {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    _child: Child,
}

impl NcnnProc {
    fn spawn(param: &str, bin: &str, in_name: &str, out_name: &str, ms: u32) -> io::Result<Self> {
        let code = format!(r#"
import sys,os,struct,numpy as np
import ncnn
ncnn.destroy_gpu_instance();ncnn.create_gpu_instance()
net=ncnn.Net();net.set_vulkan_device(ncnn.get_gpu_device(0))
o=net.opt;o.use_vulkan_compute=True;o.use_fp16_arithmetic=True
o.use_fp16_packed=True;o.use_fp16_storage=True
net.load_param("{param}");net.load_model("{bin}")
sys.stderr.write("READY\n");sys.stderr.flush()
while True:
 h=os.read(0,4)
 if not h:break
 s=struct.unpack('>I',h)[0];d=b''
 while len(d)<s:
  c=os.read(0,s-len(d))
  if not c:break
  d+=c
 f=np.frombuffer(d,dtype=np.float16).astype(np.float32).copy().reshape(1,3,{ms},{ms})*255
 with net.create_extractor() as ex:
  ex.input("{in_}",ncnn.Mat(f))
  ret,out=ex.extract("{out_}")
 raw=np.array(out).copy().tobytes()
 os.write(1,struct.pack('>I',len(raw)));os.write(1,raw)
"#, param=param, bin=bin, in_=in_name, out_=out_name, ms=ms);

        let mut child = Command::new("python3").arg("-c").arg(&code)
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;

        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());

        // Wait for READY
        let mut stderr = BufReader::new(child.stderr.take().unwrap());
        let mut line = String::new();
        let start = Instant::now();
        while start.elapsed().as_secs() < 30 {
            line.clear();
            stderr.read_line(&mut line).ok();
            if line.contains("READY") { break; }
        }
        eprintln!("NCNN_READY");
        Ok(NcnnProc { stdin, stdout, _child: child })
    }

    fn forward(&mut self, frame_f16: &[u8]) -> io::Result<Vec<f32>> {
        let size = (frame_f16.len() as u32).to_be_bytes();
        self.stdin.write_all(&size)?;
        self.stdin.write_all(frame_f16)?;
        self.stdin.flush()?;

        let mut hdr = [0u8; 4];
        self.stdout.get_mut().read_exact(&mut hdr)?;
        let n = u32::from_be_bytes(hdr) as usize;
        let mut buf = vec![0u8; n];
        self.stdout.get_mut().read_exact(&mut buf)?;

        let floats: Vec<f32> = unsafe {
            std::slice::from_raw_parts(buf.as_ptr() as *const f32, n/4).to_vec()
        };
        Ok(floats)
    }
}

// ── Rayon-parallel post-process ───────────────────────────────────

fn process_rayon(raw: &[f32], n_cells: usize, model_size: f32,
                  score_thresh: f32, nms_thresh: f32,
                  out: &mut [f32; 120]) -> usize
{
    // ncnn output layout: (84 rows × n_cells columns), row-major flat.
    // Row 0=cl, 1=cy, 2=bw, 3=bh, rows 4-83=class_scores[0..79].
    let candidates: Vec<_> = (0..n_cells).into_par_iter().filter_map(|j| {
        let cx=raw[j]; let cy=raw[n_cells+j];
        let bw=raw[2*n_cells+j]; let bh=raw[3*n_cells+j];
        let x1=(cx-bw*0.5).max(0.0); let y1=(cy-bh*0.5).max(0.0);
        let x2=(cx+bw*0.5).min(model_size); let y2=(cy+bh*0.5).min(model_size);
        if x2<=x1||y2<=y1 { return None; }
        let mut bs=0.0f32; let mut bc=0i32;
        for c in 0..80 {
            let s=raw[(4+c)*n_cells + j];
            if s>bs { bs=s; bc=c as i32; }
        }
        if bs<score_thresh { return None; }
        Some((x1,y1,x2,y2,bs,bc))
    }).collect();

    if candidates.is_empty() { return 0; }

    let mut sorted: Vec<usize> = (0..candidates.len()).collect();
    sorted.par_sort_unstable_by(|&a,&b|
        candidates[b].4.partial_cmp(&candidates[a].4).unwrap_or(std::cmp::Ordering::Equal));

    let areas: Vec<f32> = candidates.iter().map(|(x1,y1,x2,y2,_,_)| (x2-x1)*(y2-y1)).collect();
    let mut keep: Vec<usize> = Vec::with_capacity(20);
    for &i in &sorted {
        if keep.len()>=20 { break; }
        let (ax1,ay1,ax2,ay2,_,_) = candidates[i];
        if keep.iter().any(|&j| {
            let (bx1,by1,bx2,by2,_,_) = candidates[j];
            let w=(ax2.min(bx2)-ax1.max(bx1)).max(0.0);
            let h=(ay2.min(by2)-ay1.max(by1)).max(0.0);
            (w*h)/(areas[i]+areas[j]-w*h) > nms_thresh
        }) { continue; }
        keep.push(i);
    }
    // Sort keep by score descending (Frigate expects this for early-break)
    keep.sort_unstable_by(|&a, &b| candidates[b].4.partial_cmp(&candidates[a].4).unwrap_or(std::cmp::Ordering::Equal));
    for (k,&idx) in keep.iter().enumerate() {
        let (x1,y1,x2,y2,s,c) = candidates[idx];
        let o=k*6; out[o]=c as f32;out[o+1]=s;out[o+2]=x1;out[o+3]=y1;out[o+4]=x2;out[o+5]=y2;
    }
    keep.len()
}

// ── Channel types ─────────────────────────────────────────────────

struct Frame { data: Vec<u8> }
enum Msg { Frame(Arc<Frame>), Shutdown }

// ── Main ──────────────────────────────────────────────────────────

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 7 {
        eprintln!("Usage: {} <param> <bin> <in_name> <out_name> <ms> <arch>", args[0]);
        std::process::exit(1);
    }

    let ms: u32 = args[5].parse().unwrap_or(640);
    let is_yolo26 = args[6] == "yolo26";

    // Spawn ncnn subprocess
    let mut ncnn = NcnnProc::spawn(&args[1], &args[2], &args[3], &args[4], ms)?;

    // Channels
    let (tx_frame, rx_frame): (Sender<Msg>, Receiver<Msg>) = bounded(2);
    let (tx_result, rx_result): (Sender<([f32; 120], usize)>, Receiver<([f32; 120], usize)>) = bounded(2);

    // Thread 2: Inference + post-process
    std::thread::spawn(move || loop {
        let frame = match rx_frame.recv() {
            Ok(Msg::Frame(f)) => f,
            _ => break,
        };
        let raw = ncnn.forward(&frame.data).unwrap_or_default();
        let n_cells = raw.len() / if is_yolo26 { 84 } else { 144 };
        let mut out = [0.0f32; 120];
        let n = if is_yolo26 {
            process_rayon(&raw, n_cells, ms as f32, 0.05, 0.45, &mut out)
        } else { 0 };
        if tx_result.send((out, n)).is_err() { break; }
    });

    // Thread 3: stdout writer
    std::thread::spawn(move || {
        while let Ok((data, _)) = rx_result.recv() {
            let bytes: &[u8] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, 480) };
            let mut out = io::stdout();
            out.write_all(&480u32.to_be_bytes()).ok();
            out.write_all(bytes).ok();
            out.flush().ok();
        }
    });

    // Main: stdin reader
    let mut count: u64 = 0;
    let t0 = Instant::now();
    loop {
        let mut hdr = [0u8; 4];
        if io::stdin().read_exact(&mut hdr).is_err() { break; }
        let size = u32::from_be_bytes(hdr) as usize;
        let mut data = vec![0u8; size];
        if io::stdin().read_exact(&mut data).is_err() { break; }
        if tx_frame.send(Msg::Frame(Arc::new(Frame{data}))).is_err() { break; }
        count += 1;
        if count % 100 == 0 {
            eprintln!("STATS frames={} fps={:.1}", count, count as f64/t0.elapsed().as_secs_f64());
        }
    }
    drop(tx_frame);
    eprintln!("EXIT frames={}", count);
    Ok(())
}
