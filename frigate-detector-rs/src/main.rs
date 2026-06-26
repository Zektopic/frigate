//! Standalone Rust detector binary — replaces Python YOLO26 worker.
//!
//! Pipe protocol (drop-in compatible):
//!   stdin:  [4B BE size][float16 frame (1×3×640×640)]
//!   stdout: [4B BE size][float32 detections (20×6)]
//!   stderr: "READY\n" on startup

use std::io::{self, Read, Write};
use std::process::{Command, Stdio, Child, ChildStdin, ChildStdout};
use std::time::Instant;

// ── Rust post-processing (replaces numpy process_yolo26) ──────────

fn process_yolo26(raw: &[f32], n: usize, model_size: f32,
                   score_thresh: f32, nms_thresh: f32,
                   out: &mut [f32; 120]) -> usize {
    let mut boxes: Vec<(f32, f32, f32, f32)> = Vec::with_capacity(512);
    let mut scores_v: Vec<f32> = Vec::with_capacity(512);
    let mut classes: Vec<i32> = Vec::with_capacity(512);

    for i in 0..n {
        let b = i * 84;
        let cx = raw[b]; let cy = raw[b+1]; let bw = raw[b+2]; let bh = raw[b+3];
        let x1 = (cx - bw * 0.5).max(0.0);
        let y1 = (cy - bh * 0.5).max(0.0);
        let x2 = (cx + bw * 0.5).min(model_size);
        let y2 = (cy + bh * 0.5).min(model_size);
        if x2 <= x1 || y2 <= y1 { continue; }

        let mut best_c: i32 = 0;
        let mut best_s: f32 = 0.0;
        for c in 0..80 {
            let s = raw[b + 4 + c];
            if s > best_s { best_s = s; best_c = c as i32; }
        }
        if best_s < score_thresh { continue; }

        boxes.push((x1, y1, x2, y2));
        scores_v.push(best_s);
        classes.push(best_c);
    }

    let n_boxes = boxes.len();
    if n_boxes == 0 { return 0; }

    // NMS
    let areas: Vec<f32> = boxes.iter().map(|(x1,y1,x2,y2)| (x2-x1)*(y2-y1)).collect();
    let mut order: Vec<usize> = (0..n_boxes).collect();
    order.sort_unstable_by(|&a, &b| scores_v[b].partial_cmp(&scores_v[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep: Vec<usize> = Vec::with_capacity(20);
    for &i in &order {
        if keep.len() >= 20 { break; }
        let sup = keep.iter().any(|&j| {
            let (ax1,ay1,ax2,ay2) = boxes[i];
            let (bx1,by1,bx2,by2) = boxes[j];
            let xx1 = ax1.max(bx1); let yy1 = ay1.max(by1);
            let xx2 = ax2.min(bx2); let yy2 = ay2.min(by2);
            let w = (xx2 - xx1).max(0.0); let h = (yy2 - yy1).max(0.0);
            let inter = w * h;
            inter / (areas[i] + areas[j] - inter) > nms_thresh
        });
        if !sup { keep.push(i); }
    }

    let sx = 1.0; let sy = 1.0;
    for (k, &idx) in keep.iter().enumerate() {
        let (x1, y1, x2, y2) = boxes[idx];
        let o = k * 6;
        out[o] = classes[idx] as f32;
        out[o+1] = scores_v[idx];
        out[o+2] = y1 * sy; out[o+3] = x1 * sx;
        out[o+4] = y2 * sy; out[o+5] = x2 * sx;
    }
    keep.len()
}

// ── Python ncnn subprocess wrapper ────────────────────────────────

struct NcnnProc {
    stdin: ChildStdin,
    stdout: ChildStdout,
    _child: Child,
}

impl NcnnProc {
    fn spawn(param: &str, bin: &str, in_name: &str, out_name: &str, model_size: u32) -> io::Result<Self> {
        let code = format!(r#"
import sys,os,struct,numpy as np
import ncnn
ncnn.destroy_gpu_instance();ncnn.create_gpu_instance()
net=ncnn.Net();net.set_vulkan_device(ncnn.get_gpu_device(0))
net.opt.use_vulkan_compute=True;net.opt.use_fp16_arithmetic=True
net.opt.use_fp16_packed=True;net.opt.use_fp16_storage=True
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
    f=np.frombuffer(d,dtype=np.float16).astype(np.float32).copy().reshape(1,3,{ms},{ms})*255.0
    m=ncnn.Mat(f)
    with net.create_extractor() as ex:
        ex.input("{in_}",m)
        ret,out=ex.extract("{out_}")
    raw=np.array(out).copy().tobytes()
    os.write(1,struct.pack('>I',len(raw)));os.write(1,raw)
"#, param=param, bin=bin, in_=in_name, out_=out_name, ms=model_size);

        let mut child = Command::new("python3")
            .args(["-c", &code])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();

        // Wait for READY
        let mut stderr = child.stderr.take().unwrap();
        let mut buf = [0u8; 64];
        let mut ready = String::new();
        let start = Instant::now();
        while start.elapsed().as_secs() < 30 {
            let n = stderr.read(&mut buf).unwrap_or(0);
            if n > 0 {
                ready.push_str(&String::from_utf8_lossy(&buf[..n]));
                if ready.contains("READY") { break; }
            }
        }
        eprintln!("NCNN_READY");
        Ok(NcnnProc { stdin, stdout, _child: child })
    }

    fn infer(&mut self, frame_f16: &[u8]) -> io::Result<Vec<f32>> {
        let size = (frame_f16.len() as u32).to_be_bytes();
        self.stdin.write_all(&size)?;
        self.stdin.write_all(frame_f16)?;
        let mut header = [0u8; 4];
        self.stdout.read_exact(&mut header)?;
        let size = u32::from_be_bytes(header) as usize;
        let mut data = vec![0u8; size];
        self.stdout.read_exact(&mut data)?;
        let floats: Vec<f32> = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const f32,
                size / 4,
            ).to_vec()
        };
        Ok(floats)
    }
}

// ── Main ───────────────────────────────────────────────────────────

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 7 {
        eprintln!("Usage: {} <param> <bin> <in_name> <out_name> <model_size> <arch>", args[0]);
        std::process::exit(1);
    }
    let param = &args[1]; let bin = &args[2];
    let in_name = &args[3]; let out_name = &args[4];
    let model_size: u32 = args[5].parse().unwrap_or(640);
    let arch = &args[6]; // "yolo26" or "yolov8"
    let is_yolo26 = arch == "yolo26";

    // Spawn Python ncnn subprocess (GPU inference only, no post-processing)
    let mut ncnn = NcnnProc::spawn(param, bin, in_name, out_name, model_size)?;

    let mut out_buf: [f32; 120] = [0.0; 120];
    let mut frame_count: u64 = 0;
    let mut total_pp_ms: f64 = 0.0;

    loop {
        // Read frame from stdin (float16, same protocol as before)
        let mut header = [0u8; 4];
        if io::stdin().read_exact(&mut header).is_err() { break; }
        let size = u32::from_be_bytes(header) as usize;
        let mut frame_data = vec![0u8; size];
        if io::stdin().read_exact(&mut frame_data).is_err() { break; }

        // Forward to ncnn Python subprocess (GPU inference)
        let raw_output = ncnn.infer(&frame_data)?; // raw bytes → float32

        // Rust post-processing (NO numpy!)
        if is_yolo26 {
            let t0 = Instant::now();
            let n = process_yolo26(&raw_output, raw_output.len() / 84, model_size as f32, 0.05, 0.45, &mut out_buf);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            total_pp_ms += ms;

            if frame_count % 100 == 0 {
                eprintln!("FRAME {} pp={:.2}ms avg={:.2}ms n={}", frame_count, ms, total_pp_ms / (frame_count+1) as f64, n);
            }
        } else {
            // Keep raw output as-is for non-YOLO26 (rarely used)
            let n = (raw_output.len() / 6).min(120);
            for i in 0..n { out_buf[i] = raw_output[i]; }
        }

        // Write result (480 bytes = 20×6×4)
        let result_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(out_buf.as_ptr() as *const u8, 120 * 4)
        };
        let size_be = ((result_bytes.len()) as u32).to_be_bytes();
        io::stdout().write_all(&size_be)?;
        io::stdout().write_all(result_bytes)?;
        frame_count += 1;
    }

    eprintln!("EXIT frames={} total_pp={:.2}ms avg_pp={:.2}ms", frame_count, total_pp_ms, total_pp_ms / frame_count.max(1) as f64);
    Ok(())
}
