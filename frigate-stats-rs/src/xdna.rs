//! AMD XDNA (Ryzen AI) NPU telemetry from the in-tree `amdxdna` driver.
//!
//! The driver has no fdinfo keys and, on desktop APUs, no utilisation
//! sensors, so usage is the share of the array's columns owned by contexts
//! that ran commands since the previous query. Struct layouts mirror
//! include/uapi/drm/amdxdna_accel.h (kernel 7.1).
//!
//! Any query wakes a runtime-suspended NPU and restarts its idle timer, so a
//! suspended device is reported from sysfs alone, and an idle one is left
//! alone long enough to suspend.

use std::collections::{BTreeSet, HashMap};
use std::ffi::{c_int, c_ulong};
use std::fs::{self, File, OpenOptions};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

use crate::drm::read_trim;

extern "C" {
    fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
}

/// `_IOWR('d', DRM_COMMAND_BASE + nr, size)`.
const fn drm_iowr(nr: u32, size: usize) -> c_ulong {
    ((3u32 << 30) | ((size as u32) << 16) | ((b'd' as u32) << 8) | (0x40 + nr)) as c_ulong
}

const QUERY_AIE_METADATA: u32 = 1;
const QUERY_CLOCK_METADATA: u32 = 3;
const ARRAY_HW_CONTEXT_ALL: u32 = 0;
const ARRAY_BO_USAGE: u32 = 6;

#[repr(C)]
struct GetInfo {
    param: u32,
    buffer_size: u32,
    buffer: u64,
}

#[repr(C)]
struct GetArray {
    param: u32,
    element_size: u32,
    num_element: u32,
    pad: u32,
    buffer: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct TileMetadata {
    row_count: u16,
    row_start: u16,
    dma_channel_count: u16,
    lock_count: u16,
    event_reg_count: u16,
    pad: [u16; 3],
}

#[repr(C)]
#[derive(Default)]
struct AieMetadata {
    col_size: u32,
    cols: u16,
    rows: u16,
    ver_major: u32,
    ver_minor: u32,
    core: TileMetadata,
    mem: TileMetadata,
    shim: TileMetadata,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Clock {
    name: [u8; 16],
    freq_mhz: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Default)]
struct ClockMetadata {
    mp_npu: Clock,
    h: Clock,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct HwCtxEntry {
    pub context_id: u32,
    pub start_col: u32,
    pub num_col: u32,
    pub hwctx_id: u32,
    pub pid: i64,
    pub command_submissions: u64,
    pub command_completions: u64,
    pub migrations: u64,
    pub preemptions: u64,
    pub errors: u64,
    pub priority: u64,
    pub heap_usage: u64,
    pub suspensions: u64,
    pub state: u32,
    pub pasid: u32,
    pub gops: u32,
    pub fps: u32,
    pub dma_bandwidth: u32,
    pub latency: u32,
    pub frame_exec_time: u32,
    pub txn_op_idx: u32,
    pub ctx_pc: u32,
    pub fatal_error_type: u32,
    pub fatal_error_exception_type: u32,
    pub fatal_error_exception_pc: u32,
    pub fatal_error_app_module: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Default)]
struct BoUsage {
    pid: i64,
    total_usage: u64,
    internal_usage: u64,
    heap_usage: u64,
}

// Sizes measured against the running kernel.
const _: () = assert!(std::mem::size_of::<GetInfo>() == 16);
const _: () = assert!(std::mem::size_of::<GetArray>() == 24);
const _: () = assert!(std::mem::size_of::<AieMetadata>() == 64);
const _: () = assert!(std::mem::size_of::<ClockMetadata>() == 48);
const _: () = assert!(std::mem::size_of::<HwCtxEntry>() == 144);
const _: () = assert!(std::mem::size_of::<BoUsage>() == 32);

const IOCTL_GET_INFO: c_ulong = drm_iowr(7, std::mem::size_of::<GetInfo>());
const IOCTL_GET_ARRAY: c_ulong = drm_iowr(10, std::mem::size_of::<GetArray>());

#[derive(Debug, Clone, Default, PartialEq)]
pub struct NpuInfo {
    pub index: u32,
    pub name: String,
    pub pci: String,
    pub firmware: String,
    /// `/sys/class/accel/accelN`.
    pub sysfs: PathBuf,
    /// `/dev/accel/accelN`.
    pub node: PathBuf,
}

/// Finds the NPUs bound to the `amdxdna` driver.
pub fn discover(sys_root: &Path, dev_root: &Path) -> Vec<NpuInfo> {
    let Ok(entries) = fs::read_dir(sys_root.join("class/accel")) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(index) = name.strip_prefix("accel").and_then(|n| n.parse().ok()) else {
            continue;
        };
        let dev = entry.path().join("device");
        let driver = fs::read_link(dev.join("driver")).unwrap_or_default();
        if driver.file_name().is_none_or(|f| f != "amdxdna") {
            continue;
        }
        let device = read_trim(&dev.join("device")).unwrap_or_default();
        out.push(NpuInfo {
            index,
            name: match device.as_str() {
                "0x1502" => "AMD Ryzen AI NPU (XDNA1)".into(),
                "0x17f0" => "AMD Ryzen AI NPU (XDNA2)".into(),
                _ => "AMD NPU".into(),
            },
            pci: fs::canonicalize(&dev)
                .ok()
                .and_then(|p| p.file_name().map(|f| f.to_string_lossy().into_owned()))
                .unwrap_or_default(),
            firmware: read_trim(&dev.join("fw_version")).unwrap_or_default(),
            sysfs: entry.path(),
            node: dev_root.join("accel").join(&name),
        });
    }
    out.sort_by_key(|d| d.index);
    out
}

/// Where a sample's numbers came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Source {
    /// The device node is not accessible; only sysfs was read.
    #[default]
    Sysfs,
    /// The device is runtime-suspended and was not queried.
    Suspended,
    /// Idle for a while; the previous query is reused so it can suspend.
    Cached,
    /// Queried from the driver.
    Occupancy,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::Sysfs => "sysfs",
            Source::Suspended => "suspended",
            Source::Cached => "cached",
            Source::Occupancy => "occupancy",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct NpuUsage {
    pub info: NpuInfo,
    /// Whether the device node could be opened.
    pub accessible: bool,
    pub runtime_status: String,
    pub source: Source,
    /// Share of the columns owned by contexts that were busy, 0-100.
    pub util: f64,
    /// Buffer memory of the processes holding contexts, in bytes.
    pub mem_bytes: u64,
    pub contexts: u32,
    pub busy_contexts: u32,
    /// Commands completed per second since the previous query.
    pub command_rate: f64,
    pub errors: u64,
    pub h_clock_mhz: u32,
    pub mp_npu_clock_mhz: u32,
    pub columns: u16,
}

/// What the context table says about the array, given the completion
/// counters of the previous query.
#[derive(Debug, Default, PartialEq)]
pub struct Occupancy {
    pub util: f64,
    pub busy_contexts: u32,
    pub completed: u64,
    pub errors: u64,
}

/// A context is busy when it has commands in flight or completed commands
/// since the previous query.
///
/// The context's `state` is not a sign of work: every context reads as
/// active for as long as the device is awake.
pub fn occupancy(
    entries: &[HwCtxEntry],
    prev: &HashMap<(i64, u32), u64>,
    columns: usize,
) -> Occupancy {
    let columns = columns.max(1);
    let mut busy_cols = vec![false; columns];
    let mut out = Occupancy::default();
    for e in entries {
        let before = prev.get(&(e.pid, e.context_id)).copied();
        let advanced = before.is_some_and(|p| e.command_completions > p);
        if let Some(p) = before {
            out.completed += e.command_completions.saturating_sub(p);
        }
        out.errors += e.errors;
        let busy = advanced || e.command_submissions > e.command_completions;
        if busy {
            out.busy_contexts += 1;
            for col in e.start_col..e.start_col.saturating_add(e.num_col) {
                if let Some(b) = busy_cols.get_mut(col as usize) {
                    *b = true;
                }
            }
        }
    }
    out.util = 100.0 * busy_cols.iter().filter(|b| **b).count() as f64 / columns as f64;
    out
}

/// How long an idle device is left alone. Its autosuspend delay is 5 s, and
/// every query restarts that timer.
const IDLE_BACKOFF_SECS: f64 = 10.0;

/// Whether an awake device may be queried now.
///
/// While commands are running the device is awake anyway. Once a query has
/// seen no progress, the next one waits long enough for the device to
/// suspend; polled any faster, the monitor itself would keep it powered.
pub fn should_query(last_query: Option<f64>, last_progress: bool, now: f64) -> bool {
    match last_query {
        None => true,
        Some(at) => last_progress || now - at >= IDLE_BACKOFF_SECS,
    }
}

struct Device {
    info: NpuInfo,
    file: Option<File>,
    columns: u16,
    prev: HashMap<(i64, u32), u64>,
    prev_at: Option<f64>,
    /// Whether the previous query saw commands running.
    last_progress: bool,
    last_query: Option<f64>,
    last: NpuUsage,
}

impl Device {
    fn get_info<T>(&self, param: u32, buf: &mut T) -> bool {
        let Some(file) = &self.file else {
            return false;
        };
        let mut arg = GetInfo {
            param,
            buffer_size: std::mem::size_of::<T>() as u32,
            buffer: buf as *mut T as u64,
        };
        // SAFETY: `buf` is a live, writable T and buffer_size is its size.
        unsafe { ioctl(file.as_raw_fd(), IOCTL_GET_INFO, &mut arg as *mut GetInfo) == 0 }
    }

    fn get_array<T>(&self, param: u32, buf: &mut [T]) -> Option<usize> {
        let file = self.file.as_ref()?;
        let mut arg = GetArray {
            param,
            element_size: std::mem::size_of::<T>() as u32,
            num_element: buf.len() as u32,
            pad: 0,
            buffer: buf.as_mut_ptr() as u64,
        };
        // SAFETY: the buffer holds num_element elements of element_size bytes.
        let rc = unsafe { ioctl(file.as_raw_fd(), IOCTL_GET_ARRAY, &mut arg as *mut GetArray) };
        (rc == 0).then(|| (arg.num_element as usize).min(buf.len()))
    }

    fn carry_over(&self, source: Source, runtime_status: String) -> NpuUsage {
        NpuUsage {
            info: self.info.clone(),
            accessible: self.file.is_some(),
            runtime_status,
            source,
            mem_bytes: self.last.mem_bytes,
            contexts: self.last.contexts,
            columns: self.columns,
            // clocks are gated while suspended
            h_clock_mhz: if source == Source::Suspended {
                0
            } else {
                self.last.h_clock_mhz
            },
            mp_npu_clock_mhz: if source == Source::Suspended {
                0
            } else {
                self.last.mp_npu_clock_mhz
            },
            ..Default::default()
        }
    }

    fn sample(&mut self, now: f64) -> NpuUsage {
        let runtime_status =
            read_trim(&self.info.sysfs.join("device/power/runtime_status")).unwrap_or_default();

        if self.file.is_none() {
            // opening the node does not wake the device
            self.file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&self.info.node)
                .ok();
        }
        if self.file.is_none() {
            return self.carry_over(Source::Sysfs, runtime_status);
        }

        if runtime_status != "active" {
            // The counters of the last query are kept: contexts outlive a
            // suspend, so the next query still sees what ran in between.
            return self.carry_over(Source::Suspended, runtime_status);
        }
        if !should_query(self.last_query, self.last_progress, now) {
            return self.carry_over(Source::Cached, runtime_status);
        }
        self.last_query = Some(now);

        if self.columns == 0 {
            let mut md = AieMetadata::default();
            if self.get_info(QUERY_AIE_METADATA, &mut md) {
                self.columns = md.cols;
            }
        }
        let mut usage = NpuUsage {
            info: self.info.clone(),
            accessible: true,
            runtime_status,
            source: Source::Occupancy,
            columns: self.columns,
            ..Default::default()
        };
        let mut clocks = ClockMetadata::default();
        if self.get_info(QUERY_CLOCK_METADATA, &mut clocks) {
            usage.mp_npu_clock_mhz = clocks.mp_npu.freq_mhz;
            usage.h_clock_mhz = clocks.h.freq_mhz;
        }

        let mut entries = vec![HwCtxEntry::default(); 64];
        let n = self
            .get_array(ARRAY_HW_CONTEXT_ALL, &mut entries)
            .unwrap_or(0);
        let entries = &entries[..n];
        let occ = occupancy(entries, &self.prev, self.columns as usize);
        let dt = self.prev_at.map(|t| now - t).filter(|d| *d > 0.0);
        usage.util = occ.util;
        usage.contexts = n as u32;
        usage.busy_contexts = occ.busy_contexts;
        usage.errors = occ.errors;
        usage.command_rate = dt.map_or(0.0, |dt| occ.completed as f64 / dt);
        self.prev = entries
            .iter()
            .map(|e| ((e.pid, e.context_id), e.command_completions))
            .collect();
        self.prev_at = Some(now);

        let pids: BTreeSet<i64> = entries.iter().map(|e| e.pid).collect();
        for pid in pids {
            let mut bo = [BoUsage {
                pid,
                ..Default::default()
            }];
            if self.get_array(ARRAY_BO_USAGE, &mut bo).is_some() {
                usage.mem_bytes += bo[0].total_usage;
            }
        }

        self.last_progress = occ.busy_contexts > 0;
        self.last = usage.clone();
        usage
    }
}

pub struct NpuSampler {
    devices: Vec<Device>,
}

impl NpuSampler {
    pub fn new(sys_root: &Path, dev_root: &Path) -> Self {
        NpuSampler {
            devices: discover(sys_root, dev_root)
                .into_iter()
                .map(|info| Device {
                    info,
                    file: None,
                    columns: 0,
                    prev: HashMap::new(),
                    prev_at: None,
                    last_progress: false,
                    last_query: None,
                    last: NpuUsage::default(),
                })
                .collect(),
        }
    }

    /// `now` is a monotonic clock reading in seconds.
    pub fn sample(&mut self, now: f64) -> Vec<NpuUsage> {
        self.devices.iter_mut().map(|d| d.sample(now)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    #[test]
    fn ioctl_numbers_match_the_uapi() {
        // DRM_IOCTL_AMDXDNA_GET_INFO / GET_ARRAY, as printed by strace
        assert_eq!(IOCTL_GET_INFO, 0xC010_6447);
        assert_eq!(IOCTL_GET_ARRAY, 0xC018_644A);
    }

    fn ctx(
        pid: i64,
        id: u32,
        start: u32,
        cols: u32,
        sub: u64,
        done: u64,
        state: u32,
    ) -> HwCtxEntry {
        HwCtxEntry {
            pid,
            context_id: id,
            start_col: start,
            num_col: cols,
            command_submissions: sub,
            command_completions: done,
            state,
            ..Default::default()
        }
    }

    #[test]
    fn idle_contexts_occupy_nothing() {
        let entries = [ctx(1, 1, 1, 4, 10, 10, 0)];
        let prev = HashMap::from([((1, 1), 10)]);
        let occ = occupancy(&entries, &prev, 5);
        assert_eq!(occ.util, 0.0);
        assert_eq!(occ.busy_contexts, 0);
        assert_eq!(occ.completed, 0);
    }

    #[test]
    fn progress_since_the_previous_query_marks_columns_busy() {
        // 4 of 5 columns; column 0 is never assigned to user contexts
        let entries = [ctx(1, 1, 1, 4, 250, 250, 0)];
        let prev = HashMap::from([((1, 1), 100)]);
        let occ = occupancy(&entries, &prev, 5);
        assert_eq!(occ.util, 80.0);
        assert_eq!(occ.busy_contexts, 1);
        assert_eq!(occ.completed, 150);
    }

    #[test]
    fn in_flight_contexts_are_busy_without_history() {
        let in_flight = [ctx(1, 1, 1, 1, 5, 4, 0)];
        assert_eq!(occupancy(&in_flight, &HashMap::new(), 5).util, 20.0);
        // a first sighting of an idle context is not progress
        let idle = [ctx(1, 1, 1, 1, 7, 7, 0)];
        let occ = occupancy(&idle, &HashMap::new(), 5);
        assert_eq!((occ.util, occ.completed), (0.0, 0));
    }

    #[test]
    fn a_loaded_context_that_ran_nothing_is_not_busy() {
        // Seen on an 8600G: while the device is awake all three contexts of
        // an idle service report state 1, with no commands running.
        let entries = [ctx(1, 1, 1, 4, 229, 229, 1)];
        let prev = HashMap::from([((1, 1), 229)]);
        let occ = occupancy(&entries, &prev, 5);
        assert_eq!((occ.util, occ.busy_contexts), (0.0, 0));
    }

    #[test]
    fn an_idle_device_is_given_time_to_suspend() {
        // never queried
        assert!(should_query(None, false, 0.0));
        // commands were running: keep following them at any rate
        assert!(should_query(Some(100.0), true, 101.0));
        // nothing ran: stay away for longer than the 5 s autosuspend delay
        assert!(!should_query(Some(100.0), false, 103.0));
        assert!(!should_query(Some(100.0), false, 109.9));
        assert!(should_query(Some(100.0), false, 110.0));
    }

    #[test]
    fn shared_and_out_of_range_columns_are_counted_once() {
        let entries = [
            ctx(1, 1, 1, 1, 2, 1, 0),
            ctx(2, 1, 1, 1, 2, 1, 0),        // shares column 1
            ctx(3, 1, 4, 9, 2, 1, 0),        // runs past the array
            ctx(4, 1, u32::MAX, 2, 2, 1, 0), // must not overflow
        ];
        let occ = occupancy(&entries, &HashMap::new(), 5);
        assert_eq!(occ.util, 40.0);
        assert_eq!(occ.busy_contexts, 4);
        // a zero column count must not divide by zero
        assert!(occupancy(&entries, &HashMap::new(), 0).util.is_finite());
    }

    fn fake_npu(dir: &TempDir, status: &str) {
        let pci = "sys/devices/pci0000:00/0000:10:00.1";
        dir.mkdir("sys/bus/pci/drivers/amdxdna");
        dir.symlink("sys/bus/pci/drivers/amdxdna", &format!("{pci}/driver"));
        dir.write(&format!("{pci}/device"), "0x1502\n");
        dir.write(&format!("{pci}/fw_version"), "1.5.5.391\n");
        dir.write(
            &format!("{pci}/power/runtime_status"),
            &format!("{status}\n"),
        );
        dir.symlink(pci, "sys/class/accel/accel0/device");
    }

    #[test]
    fn discovers_amdxdna_devices_only() {
        let dir = TempDir::new("xdna");
        fake_npu(&dir, "suspended");
        // another accelerator driver is not ours
        dir.mkdir("sys/devices/other");
        dir.mkdir("sys/bus/pci/drivers/habanalabs");
        dir.symlink("sys/bus/pci/drivers/habanalabs", "sys/devices/other/driver");
        dir.symlink("sys/devices/other", "sys/class/accel/accel1/device");

        let found = discover(&dir.path("sys"), &dir.path("dev"));
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].index, 0);
        assert_eq!(found[0].name, "AMD Ryzen AI NPU (XDNA1)");
        assert_eq!(found[0].pci, "0000:10:00.1");
        assert_eq!(found[0].firmware, "1.5.5.391");
        assert_eq!(found[0].node, dir.path("dev/accel/accel0"));
    }

    #[test]
    fn without_a_device_node_only_sysfs_is_reported() {
        let dir = TempDir::new("xdna-nonode");
        fake_npu(&dir, "active");
        let mut sampler = NpuSampler::new(&dir.path("sys"), &dir.path("dev"));
        let usage = &sampler.sample(0.0)[0];
        assert!(!usage.accessible);
        assert_eq!(usage.source, Source::Sysfs);
        assert_eq!(usage.runtime_status, "active");
        assert_eq!(usage.util, 0.0);
    }

    #[test]
    fn a_suspended_device_is_not_queried() {
        let dir = TempDir::new("xdna-suspended");
        fake_npu(&dir, "suspended");
        // a plain file stands in for the node: an ioctl on it would fail,
        // and the suspended path must not get that far
        dir.write("dev/accel/accel0", "");
        let mut sampler = NpuSampler::new(&dir.path("sys"), &dir.path("dev"));
        let usage = &sampler.sample(0.0)[0];
        assert!(usage.accessible);
        assert_eq!(usage.source, Source::Suspended);
        assert_eq!(usage.h_clock_mhz, 0);
    }

    #[test]
    fn no_accel_class_means_no_devices() {
        let dir = TempDir::new("xdna-none");
        let mut sampler = NpuSampler::new(&dir.path("sys"), &dir.path("dev"));
        assert!(sampler.sample(0.0).is_empty());
    }
}
