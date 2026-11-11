//! Resource monitor for Frigate: per-process CPU and memory, GPU and NPU
//! usage, collected straight from procfs, sysfs and the DRM drivers.
//!
//! Python calls `frigate_stats_collect` once per stats interval and gets a
//! JSON document back. The counters of the previous call are kept here, so
//! usage is always measured over the whole interval between two calls rather
//! than over a short sampling window.

mod drm;
mod json;
mod procs;
#[cfg(test)]
mod testutil;
mod xdna;

use std::ffi::{c_char, CStr};
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use json::Json;

/// Bumped when the JSON layout changes incompatibly.
pub const ABI_VERSION: u32 = 1;

extern "C" {
    fn sysconf(name: i32) -> i64;
}
// Linux values, identical on glibc and musl for every architecture.
const SC_CLK_TCK: i32 = 2;
const SC_PAGESIZE: i32 = 30;

fn sysconf_or(name: i32, default: u64) -> u64 {
    // SAFETY: sysconf has no preconditions.
    let v = unsafe { sysconf(name) };
    if v > 0 {
        v as u64
    } else {
        default
    }
}

pub struct Collector {
    started: Instant,
    procs: procs::ProcSampler,
    gpus: drm::GpuSampler,
    npus: xdna::NpuSampler,
}

impl Collector {
    /// A collector reading the given directories in place of `/proc`, `/sys`,
    /// `/dev` and `/sys/fs/cgroup`.
    pub fn with_roots(
        proc_root: &Path,
        sys_root: &Path,
        dev_root: &Path,
        cgroup_root: &Path,
        clk_tck: u64,
        page_size: u64,
    ) -> Self {
        Collector {
            started: Instant::now(),
            procs: procs::ProcSampler::new(proc_root, cgroup_root, clk_tck, page_size),
            gpus: drm::GpuSampler::new(proc_root, sys_root, dev_root),
            npus: xdna::NpuSampler::new(sys_root, dev_root),
        }
    }

    pub fn system() -> Self {
        let c = Self::with_roots(
            Path::new("/proc"),
            Path::new("/sys"),
            Path::new("/dev"),
            Path::new("/sys/fs/cgroup"),
            sysconf_or(SC_CLK_TCK, 100),
            sysconf_or(SC_PAGESIZE, 4096),
        );
        c.gpus.start_background();
        c
    }

    pub fn collect(&mut self, root_pid: i32, extra_names: &[String]) -> String {
        let now = self.started.elapsed().as_secs_f64();
        self.collect_at(root_pid, extra_names, now)
    }

    /// `now` is a monotonic clock reading in seconds.
    pub fn collect_at(&mut self, root_pid: i32, extra_names: &[String], now: f64) -> String {
        let procs = self.procs.sample(root_pid, extra_names, now);
        let pids: Vec<i32> = procs.procs.iter().map(|p| p.pid).collect();
        let gpus = self.gpus.sample(&pids, now);
        let npus = self.npus.sample(now);

        let mut j = Json::new();
        j.begin_object();
        j.field_uint("version", ABI_VERSION as u64);

        let sys = &procs.system;
        j.key("system").begin_object();
        j.field_float("cpu", sys.cpu, 1);
        j.field_float("mem", sys.mem, 1);
        j.field_uint("mem_total_kb", sys.mem_total_kb);
        j.field_uint("mem_available_kb", sys.mem_available_kb);
        j.field_opt_uint("mem_limit_kb", sys.mem_limit_kb);
        j.field_uint("cpus", sys.cpus as u64);
        j.end_object();

        j.key("processes").begin_object();
        for p in &procs.procs {
            j.key(&p.pid.to_string()).begin_object();
            j.field_int("ppid", p.ppid as i64);
            j.field_str("name", &p.comm);
            j.field_str("cmdline", &p.cmdline);
            j.field_uint("threads", p.threads as u64);
            j.field_float("cpu", p.cpu, 1);
            j.field_float("cpu_average", p.cpu_average, 1);
            j.field_float("mem", p.mem, 1);
            j.field_uint("rss_kb", p.rss_kb);
            j.field_float("tree_cpu", p.tree_cpu, 1);
            j.field_float("tree_mem", p.tree_mem, 1);
            j.field_uint("tree_rss_kb", p.tree_rss_kb);
            j.end_object();
        }
        j.end_object();

        j.key("gpus").begin_array();
        for g in &gpus {
            j.begin_object();
            j.field_str("card", &g.card.card);
            j.field_str("pdev", &g.card.pdev);
            j.field_str("driver", &g.card.driver);
            j.field_str("vendor", &g.card.vendor);
            j.field_str("device", &g.card.device);
            if let Some(node) = &g.card.render_node {
                j.field_str("render_node", node);
            }
            j.field_bool("node_present", g.card.node_present);
            if let Some(status) = &g.runtime_status {
                j.field_str("runtime_status", status);
            }
            j.field_opt_float("busy", g.busy, 2);
            j.field_uint("busy_samples", g.busy_samples as u64);
            j.field_opt_uint("vram_used", g.vram_used);
            j.field_opt_uint("vram_total", g.vram_total);
            j.field_opt_uint("gtt_used", g.gtt_used);
            j.field_opt_uint("gtt_total", g.gtt_total);
            j.field_opt_float("temp", g.temp, 1);
            j.field_opt_float("power", g.power, 2);
            j.field_opt_float("freq_mhz", g.freq_mhz, 0);
            j.field_bool("has_decode_engine", g.has_decode_engine);
            j.key("engines").begin_object();
            for (engine, pct) in &g.engines {
                j.field_float(engine, *pct, 2);
            }
            j.end_object();
            j.key("clients").begin_object();
            for c in &g.clients {
                j.key(&c.pid.to_string()).begin_object();
                j.field_uint("vram_kb", c.vram_kb);
                j.field_uint("gtt_kb", c.gtt_kb);
                j.key("engines").begin_object();
                for (engine, pct) in &c.engines {
                    j.field_float(engine, *pct, 2);
                }
                j.end_object();
                j.end_object();
            }
            j.end_object();
            j.end_object();
        }
        j.end_array();

        j.key("npus").begin_array();
        for n in &npus {
            j.begin_object();
            j.field_str("driver", "amdxdna");
            j.field_str("name", &n.info.name);
            j.field_str("pdev", &n.info.pci);
            j.field_str("firmware", &n.info.firmware);
            j.field_bool("accessible", n.accessible);
            j.field_str("runtime_status", &n.runtime_status);
            j.field_str("source", n.source.as_str());
            j.field_float("util", n.util, 1);
            j.field_uint("mem_bytes", n.mem_bytes);
            j.field_uint("contexts", n.contexts as u64);
            j.field_uint("busy_contexts", n.busy_contexts as u64);
            j.field_float("command_rate", n.command_rate, 1);
            j.field_uint("errors", n.errors);
            j.field_uint("h_clock_mhz", n.h_clock_mhz as u64);
            j.field_uint("mp_npu_clock_mhz", n.mp_npu_clock_mhz as u64);
            j.field_uint("columns", n.columns as u64);
            j.end_object();
        }
        j.end_array();

        j.end_object();
        j.finish()
    }
}

static COLLECTOR: Mutex<Option<Collector>> = Mutex::new(None);
/// The most recent document, for a caller whose buffer was too small.
static LAST: Mutex<String> = Mutex::new(String::new());

/// Copies `doc` to `out` if it fits in `cap` bytes, and returns its length.
///
/// # Safety
///
/// `out` must be null or valid for writes of `cap` bytes.
unsafe fn copy_out(doc: &str, out: *mut u8, cap: usize) -> i64 {
    if !out.is_null() && doc.len() <= cap {
        // SAFETY: the caller guarantees `out` holds `cap` >= doc.len() bytes.
        unsafe { std::ptr::copy_nonoverlapping(doc.as_ptr(), out, doc.len()) };
    }
    doc.len() as i64
}

#[no_mangle]
pub extern "C" fn frigate_stats_abi_version() -> u32 {
    ABI_VERSION
}

/// Collects usage of the process tree under `root_pid` (0: the calling
/// process) and of the GPUs and NPUs it uses, as a JSON document.
///
/// `extra_names` is an optional comma-separated list; processes whose name
/// contains one of the entries are included with their descendants.
///
/// Returns the length of the document. It is written to `out` only if it
/// fits in `cap` bytes; otherwise fetch it with `frigate_stats_last` and a
/// larger buffer. (Collecting again instead would measure usage over the
/// few microseconds since this call.) Returns -1 on an internal error.
///
/// # Safety
///
/// `out` must be valid for writes of `cap` bytes, and `extra_names` must be
/// null or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn frigate_stats_collect(
    root_pid: i32,
    extra_names: *const c_char,
    out: *mut u8,
    cap: usize,
) -> i64 {
    let extras: Vec<String> = if extra_names.is_null() {
        Vec::new()
    } else {
        // SAFETY: the caller passes a NUL-terminated string.
        unsafe { CStr::from_ptr(extra_names) }
            .to_string_lossy()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    };
    let root = if root_pid > 0 {
        root_pid
    } else {
        std::process::id() as i32
    };

    let doc = std::panic::catch_unwind(|| {
        let mut guard = COLLECTOR.lock().unwrap_or_else(|e| e.into_inner());
        guard
            .get_or_insert_with(Collector::system)
            .collect(root, &extras)
    });
    let Ok(doc) = doc else {
        return -1;
    };
    // SAFETY: forwarded from the caller.
    let len = unsafe { copy_out(&doc, out, cap) };
    *LAST.lock().unwrap_or_else(|e| e.into_inner()) = doc;
    len
}

/// Copies the document of the most recent `frigate_stats_collect` call,
/// without collecting again. Returns its length (0 before the first call);
/// it is written only if it fits in `cap` bytes.
///
/// # Safety
///
/// `out` must be null or valid for writes of `cap` bytes.
#[no_mangle]
pub unsafe extern "C" fn frigate_stats_last(out: *mut u8, cap: usize) -> i64 {
    let last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: forwarded from the caller.
    unsafe { copy_out(&last, out, cap) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    fn empty_collector(dir: &TempDir) -> Collector {
        Collector::with_roots(
            &dir.path("proc"),
            &dir.path("sys"),
            &dir.path("dev"),
            &dir.path("cgroup"),
            100,
            4096,
        )
    }

    #[test]
    fn document_for_a_machine_without_gpu_or_npu() {
        let dir = TempDir::new("lib");
        dir.write("proc/uptime", "100.00 0\n");
        dir.write("proc/stat", "cpu  10 0 0 90 0 0 0 0 0 0\ncpu0 1 1 1 1\n");
        dir.write("proc/meminfo", "MemTotal: 1000 kB\nMemAvailable: 500 kB\n");
        dir.write(
            "proc/7/stat",
            "7 (py \"x\") S 1 0 0 0 -1 0 0 0 0 0 500 0 0 0 20 0 2 0 0 0 10 0\n",
        );
        dir.write("proc/7/cmdline", "python3\0-m\0frigate\0");
        let mut c = empty_collector(&dir);
        let doc = c.collect_at(7, &[], 0.0);
        assert_eq!(
            doc,
            "{\"version\":1,\
             \"system\":{\"cpu\":10.0,\"mem\":50.0,\"mem_total_kb\":1000,\
             \"mem_available_kb\":500,\"cpus\":1},\
             \"processes\":{\"7\":{\"ppid\":1,\"name\":\"py \\\"x\\\"\",\
             \"cmdline\":\"python3 -m frigate\",\"threads\":2,\"cpu\":5.0,\
             \"cpu_average\":5.0,\"mem\":4.0,\"rss_kb\":40,\"tree_cpu\":5.0,\
             \"tree_mem\":4.0,\"tree_rss_kb\":40}},\
             \"gpus\":[],\"npus\":[]}"
        );
    }

    #[test]
    fn ffi_reports_the_length_and_respects_the_buffer() {
        let mut small = [0u8; 4];
        // SAFETY: valid buffer and a null name list.
        let need =
            unsafe { frigate_stats_collect(0, std::ptr::null(), small.as_mut_ptr(), small.len()) };
        assert!(need > 4, "a real document is longer than 4 bytes");
        assert_eq!(small, [0u8; 4], "nothing is written when it does not fit");

        // the document that did not fit can be fetched without re-collecting
        let mut again = vec![0u8; need as usize];
        // SAFETY: valid buffer of the reported length.
        let got = unsafe { frigate_stats_last(again.as_mut_ptr(), again.len()) };
        assert_eq!(got, need);
        assert!(again.starts_with(b"{\"version\":1,") && again.ends_with(b"}"));
        // SAFETY: a null buffer only asks for the length.
        assert_eq!(unsafe { frigate_stats_last(std::ptr::null_mut(), 0) }, need);

        let mut buf = vec![0u8; need as usize + 65536];
        let names = c"go2rtc, certsync,";
        // SAFETY: valid buffer and NUL-terminated names.
        let len = unsafe { frigate_stats_collect(0, names.as_ptr(), buf.as_mut_ptr(), buf.len()) };
        assert!(len > 0 && (len as usize) <= buf.len());
        let doc = std::str::from_utf8(&buf[..len as usize]).unwrap();
        assert!(doc.starts_with("{\"version\":1,\"system\":{"));
        assert!(doc.ends_with('}'));
        // this test process is the root, so it is in the document
        assert!(doc.contains(&format!("\"{}\":{{\"ppid\":", std::process::id())));
    }

    #[test]
    fn abi_version_is_exported() {
        assert_eq!(frigate_stats_abi_version(), ABI_VERSION);
    }
}
