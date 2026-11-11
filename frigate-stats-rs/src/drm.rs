//! GPU usage from the kernel: sysfs for device-wide load, memory and sensors,
//! and DRM fdinfo (`/proc/<pid>/fdinfo/<fd>`) for per-process engine time.
//!
//! This replaces polling `radeontop` in a subprocess for AMD GPUs. fdinfo
//! reports cumulative nanoseconds per engine and DRM client, so the share of
//! an engine a process used is the difference between two samples divided by
//! the time between them.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub fn read_trim(path: &Path) -> Option<String> {
    fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

pub fn read_u64(path: &Path) -> Option<u64> {
    read_trim(path)?.parse().ok()
}

/// A file name of a symlink's target, e.g. the driver bound to a device.
fn link_name(path: &Path) -> Option<String> {
    let target = fs::read_link(path).ok()?;
    Some(target.file_name()?.to_string_lossy().into_owned())
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuCard {
    /// DRM card name, e.g. `card1`.
    pub card: String,
    /// PCI address, e.g. `0000:0f:00.0`. Matches `drm-pdev` in fdinfo.
    pub pdev: String,
    pub driver: String,
    pub vendor: String,
    pub device: String,
    /// Render node name, e.g. `renderD128`.
    pub render_node: Option<String>,
    /// Whether a device node of this GPU exists under /dev/dri. sysfs shows
    /// the host's hardware even when the device was not passed into the
    /// container, so this is what tells a usable GPU from a visible one.
    pub node_present: bool,
    /// `/sys/class/drm/<card>/device`.
    pub sysfs: PathBuf,
}

/// Finds the GPUs known to DRM.
pub fn discover(sys_root: &Path, dev_root: &Path) -> Vec<GpuCard> {
    let drm = sys_root.join("class/drm");
    let Ok(entries) = fs::read_dir(&drm) else {
        return Vec::new();
    };
    let names: Vec<String> = entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    let is_node = |name: &str, prefix: &str| {
        name.strip_prefix(prefix)
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
    };

    let mut cards = Vec::new();
    for name in names.iter().filter(|n| is_node(n, "card")) {
        let sysfs = drm.join(name).join("device");
        let Ok(real) = fs::canonicalize(&sysfs) else {
            continue;
        };
        let render_node = names
            .iter()
            .filter(|n| is_node(n, "renderD"))
            .find(|n| fs::canonicalize(drm.join(n).join("device")).is_ok_and(|p| p == real))
            .cloned();
        let dri = dev_root.join("dri");
        let node_present =
            dri.join(name).exists() || render_node.as_ref().is_some_and(|r| dri.join(r).exists());
        cards.push(GpuCard {
            card: name.clone(),
            pdev: real
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_default(),
            driver: link_name(&sysfs.join("driver")).unwrap_or_default(),
            vendor: read_trim(&sysfs.join("vendor")).unwrap_or_default(),
            device: read_trim(&sysfs.join("device")).unwrap_or_default(),
            render_node,
            node_present,
            sysfs,
        });
    }
    cards.sort_by(|a, b| a.card.cmp(&b.card));
    cards
}

/// One DRM client as described by an fdinfo file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FdInfo {
    pub driver: String,
    pub client_id: u64,
    pub pdev: String,
    /// Engine name -> cumulative busy time in nanoseconds.
    pub engines: BTreeMap<String, u64>,
    pub vram_kb: u64,
    pub gtt_kb: u64,
}

/// Parses a size such as `84436 KiB` into KiB. A bare number is bytes.
fn parse_kib(value: &str) -> Option<u64> {
    let mut parts = value.split_ascii_whitespace();
    let n: u64 = parts.next()?.parse().ok()?;
    Some(match parts.next() {
        Some("KiB") => n,
        Some("MiB") => n * 1024,
        Some("GiB") => n * 1024 * 1024,
        _ => n / 1024,
    })
}

/// Parses the DRM keys of an fdinfo file. Returns `None` for a file that is
/// not a DRM client.
pub fn parse_fdinfo(text: &str) -> Option<FdInfo> {
    let mut info = FdInfo::default();
    let mut has_client = false;
    // drm-memory-* is the long-standing key; newer kernels also print
    // drm-resident-*, which is only used when the former is missing.
    let mut resident_vram = None;
    let mut resident_gtt = None;
    let mut memory_vram = None;
    let mut memory_gtt = None;

    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();
        match key {
            "drm-driver" => info.driver = value.to_string(),
            "drm-pdev" => info.pdev = value.to_string(),
            "drm-client-id" => {
                info.client_id = value.parse().ok()?;
                has_client = true;
            }
            "drm-memory-vram" => memory_vram = parse_kib(value),
            "drm-memory-gtt" => memory_gtt = parse_kib(value),
            "drm-resident-vram" => resident_vram = parse_kib(value),
            "drm-resident-gtt" => resident_gtt = parse_kib(value),
            _ => {
                let Some(engine) = key.strip_prefix("drm-engine-") else {
                    continue;
                };
                // drm-engine-capacity-<name> is a count, not a time
                if engine.starts_with("capacity-") {
                    continue;
                }
                let Some(ns) = value.strip_suffix("ns") else {
                    continue;
                };
                if let Ok(ns) = ns.trim().parse::<u64>() {
                    info.engines.insert(engine.to_string(), ns);
                }
            }
        }
    }

    if info.driver.is_empty() || !has_client {
        return None;
    }
    info.vram_kb = memory_vram.or(resident_vram).unwrap_or(0);
    info.gtt_kb = memory_gtt.or(resident_gtt).unwrap_or(0);
    Some(info)
}

/// Reads the DRM clients held open by `pids`.
///
/// A client shows up once per file descriptor that refers to it (and in every
/// process that inherited the descriptor), so clients are de-duplicated by
/// device and client id and attributed to the first process they are seen in.
pub fn scan_clients(proc_root: &Path, pids: &[i32]) -> Vec<(i32, FdInfo)> {
    let mut seen: HashSet<(String, u64)> = HashSet::new();
    let mut clients = Vec::new();
    for &pid in pids {
        let fd_dir = proc_root.join(format!("{pid}/fd"));
        let Ok(entries) = fs::read_dir(&fd_dir) else {
            continue; // exited, or not ours to read
        };
        for entry in entries.flatten() {
            // readlink is cheaper than reading fdinfo for every descriptor
            let Ok(target) = fs::read_link(entry.path()) else {
                continue;
            };
            if !target.starts_with("/dev/dri") {
                continue;
            }
            let fdinfo = proc_root
                .join(format!("{pid}/fdinfo"))
                .join(entry.file_name());
            let Some(info) = fs::read_to_string(fdinfo)
                .ok()
                .and_then(|t| parse_fdinfo(&t))
            else {
                continue;
            };
            if seen.insert((info.pdev.clone(), info.client_id)) {
                clients.push((pid, info));
            }
        }
    }
    clients
}

/// Averages the device-wide load between two collections.
///
/// `gpu_busy_percent` is an instantaneous reading, and the stats are only
/// collected every several seconds, so a single reading says little. A
/// background thread reads it once a second and the collector takes the mean.
pub struct BusyAverager {
    /// `/sys/class/drm/<card>/device` of every sampled card.
    devices: Vec<PathBuf>,
    /// (sum, count) per device since the last `take`.
    acc: Mutex<Vec<(f64, u32)>>,
    stop: AtomicBool,
}

impl BusyAverager {
    pub fn new(devices: Vec<PathBuf>) -> Arc<Self> {
        let acc = Mutex::new(vec![(0.0, 0); devices.len()]);
        Arc::new(BusyAverager {
            devices,
            acc,
            stop: AtomicBool::new(false),
        })
    }

    /// Reads the current load of a device, without waking it up: reading a
    /// power-managed attribute resumes a runtime-suspended GPU.
    pub fn read_busy(device: &Path) -> Option<f64> {
        if is_suspended(device) {
            return None;
        }
        read_u64(&device.join("gpu_busy_percent")).map(|v| (v as f64).clamp(0.0, 100.0))
    }

    /// Takes one reading of every device.
    pub fn tick(&self) {
        let readings: Vec<Option<f64>> = self.devices.iter().map(|d| Self::read_busy(d)).collect();
        let mut acc = self.acc.lock().unwrap_or_else(|e| e.into_inner());
        for (slot, reading) in acc.iter_mut().zip(readings) {
            if let Some(v) = reading {
                slot.0 += v;
                slot.1 += 1;
            }
        }
    }

    /// Returns the mean and number of readings since the last call.
    pub fn take(&self, index: usize) -> Option<(f64, u32)> {
        let mut acc = self.acc.lock().unwrap_or_else(|e| e.into_inner());
        let slot = acc.get_mut(index)?;
        let (sum, n) = std::mem::replace(slot, (0.0, 0));
        (n > 0).then(|| (sum / n as f64, n))
    }

    pub fn spawn(self: &Arc<Self>) {
        let this = Arc::clone(self);
        let _ = std::thread::Builder::new()
            .name("frigate_gpu_busy".into())
            .spawn(move || {
                while !this.stop.load(Ordering::Relaxed) {
                    this.tick();
                    std::thread::sleep(Duration::from_secs(1));
                }
            });
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

fn is_suspended(device: &Path) -> bool {
    read_trim(&device.join("power/runtime_status"))
        .is_some_and(|s| s == "suspended" || s == "suspending")
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuClient {
    pub pid: i32,
    /// Engine name -> share of that engine used since the previous sample.
    pub engines: BTreeMap<String, f64>,
    pub vram_kb: u64,
    pub gtt_kb: u64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuUsage {
    pub card: GpuCard,
    pub runtime_status: Option<String>,
    /// Mean device load since the previous sample, 0-100.
    pub busy: Option<f64>,
    /// How many readings `busy` is the mean of.
    pub busy_samples: u32,
    pub vram_used: Option<u64>,
    pub vram_total: Option<u64>,
    pub gtt_used: Option<u64>,
    pub gtt_total: Option<u64>,
    pub temp: Option<f64>,
    pub power: Option<f64>,
    pub freq_mhz: Option<f64>,
    /// Engine name -> share used by the sampled processes, 0-100.
    pub engines: BTreeMap<String, f64>,
    /// Whether any sampled process has ever reported time on a dedicated
    /// decode engine. Newer AMD video cores have one queue for both, and
    /// account decode as `enc`.
    pub has_decode_engine: bool,
    /// Per process, sorted by pid.
    pub clients: Vec<GpuClient>,
}

pub struct GpuSampler {
    proc_root: PathBuf,
    cards: Vec<GpuCard>,
    averager: Arc<BusyAverager>,
    /// (pdev, client id) -> engine -> cumulative ns at the previous sample.
    prev: HashMap<(String, u64), BTreeMap<String, u64>>,
    prev_at: Option<f64>,
    seen_decode: HashSet<String>,
}

/// Drivers whose sysfs and fdinfo layout this module understands.
fn is_supported(card: &GpuCard) -> bool {
    card.driver == "amdgpu"
}

impl GpuSampler {
    pub fn new(proc_root: &Path, sys_root: &Path, dev_root: &Path) -> Self {
        let cards: Vec<GpuCard> = discover(sys_root, dev_root)
            .into_iter()
            .filter(is_supported)
            .collect();
        let averager = BusyAverager::new(cards.iter().map(|c| c.sysfs.clone()).collect());
        GpuSampler {
            proc_root: proc_root.to_path_buf(),
            cards,
            averager,
            prev: HashMap::new(),
            prev_at: None,
            seen_decode: HashSet::new(),
        }
    }

    pub fn has_cards(&self) -> bool {
        !self.cards.is_empty()
    }

    /// Starts the once-a-second load sampler. Not used in tests, which call
    /// `averager().tick()` themselves.
    pub fn start_background(&self) {
        if self.has_cards() {
            self.averager.spawn();
        }
    }

    #[cfg(test)]
    pub fn averager(&self) -> &Arc<BusyAverager> {
        &self.averager
    }

    /// `pids` are the processes whose GPU clients are attributed; `now` is a
    /// monotonic clock reading in seconds.
    pub fn sample(&mut self, pids: &[i32], now: f64) -> Vec<GpuUsage> {
        if self.cards.is_empty() {
            return Vec::new();
        }
        let dt_ns = self.prev_at.map(|t| (now - t) * 1e9).filter(|d| *d > 0.0);

        let clients = scan_clients(&self.proc_root, pids);
        let mut next_prev = HashMap::with_capacity(clients.len());
        // pdev -> pid -> usage
        let mut per_card: HashMap<String, BTreeMap<i32, GpuClient>> = HashMap::new();
        for (pid, info) in clients {
            let key = (info.pdev.clone(), info.client_id);
            if info.engines.contains_key("dec") {
                self.seen_decode.insert(info.pdev.clone());
            }
            let entry = per_card
                .entry(info.pdev.clone())
                .or_default()
                .entry(pid)
                .or_insert_with(|| GpuClient {
                    pid,
                    ..Default::default()
                });
            entry.vram_kb += info.vram_kb;
            entry.gtt_kb += info.gtt_kb;
            if let Some(dt_ns) = dt_ns {
                // A client that was not there last time was created since,
                // so all of its time falls into this interval.
                let before = self.prev.get(&key);
                for (engine, &ns) in &info.engines {
                    let was = before.and_then(|b| b.get(engine)).copied().unwrap_or(0);
                    let pct = 100.0 * ns.saturating_sub(was) as f64 / dt_ns;
                    *entry.engines.entry(engine.clone()).or_insert(0.0) += pct;
                }
            }
            next_prev.insert(key, info.engines);
        }
        self.prev = next_prev;
        self.prev_at = Some(now);

        self.cards
            .iter()
            .enumerate()
            .map(|(index, card)| {
                let dev = &card.sysfs;
                let runtime_status = read_trim(&dev.join("power/runtime_status"));
                let awake = !is_suspended(dev);
                let mean = self.averager.take(index);
                let busy = mean
                    .map(|(v, _)| v)
                    .or_else(|| BusyAverager::read_busy(dev));
                let sensor = |name: &str| -> Option<u64> {
                    if !awake {
                        return None;
                    }
                    let hwmon = fs::read_dir(dev.join("hwmon")).ok()?;
                    hwmon.flatten().find_map(|h| read_u64(&h.path().join(name)))
                };
                let mem = |name: &str| awake.then(|| read_u64(&dev.join(name))).flatten();

                let clients: Vec<GpuClient> = per_card
                    .remove(&card.pdev)
                    .map(|m| m.into_values().collect())
                    .unwrap_or_default();
                let mut engines: BTreeMap<String, f64> = BTreeMap::new();
                for client in &clients {
                    for (engine, pct) in &client.engines {
                        *engines.entry(engine.clone()).or_insert(0.0) += pct;
                    }
                }
                for pct in engines.values_mut() {
                    *pct = pct.clamp(0.0, 100.0);
                }

                GpuUsage {
                    card: card.clone(),
                    runtime_status,
                    busy,
                    busy_samples: mean.map_or(0, |(_, n)| n),
                    vram_used: mem("mem_info_vram_used"),
                    vram_total: mem("mem_info_vram_total"),
                    gtt_used: mem("mem_info_gtt_used"),
                    gtt_total: mem("mem_info_gtt_total"),
                    temp: sensor("temp1_input").map(|v| v as f64 / 1000.0),
                    power: sensor("power1_average")
                        .or_else(|| sensor("power1_input"))
                        .map(|v| v as f64 / 1e6),
                    freq_mhz: sensor("freq1_input").map(|v| v as f64 / 1e6),
                    engines,
                    has_decode_engine: self.seen_decode.contains(&card.pdev),
                    clients,
                }
            })
            .collect()
    }
}

impl Drop for GpuSampler {
    fn drop(&mut self) {
        self.averager.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    /// The fdinfo of an ffmpeg VAAPI decode on a Radeon 760M, kernel 7.1.
    const AMDGPU_FDINFO: &str = "pos:\t0\nflags:\t0100002\nmnt_id:\t1295\nino:\t12\n\
        drm-driver:\tamdgpu\ndrm-client-id:\t239346\ndrm-pdev:\t0000:0f:00.0\npasid:\t42752\n\
        drm-total-vram:\t84436 KiB\ndrm-resident-vram:\t84000 KiB\n\
        drm-memory-vram:\t84436 KiB\ndrm-memory-gtt: \t26568 KiB\ndrm-memory-cpu: \t0 KiB\n\
        amd-evicted-vram:\t0 KiB\ndrm-engine-gfx:\t809494162 ns\ndrm-engine-enc:\t1062490355 ns\n";

    #[test]
    fn parses_amdgpu_fdinfo() {
        let info = parse_fdinfo(AMDGPU_FDINFO).unwrap();
        assert_eq!(info.driver, "amdgpu");
        assert_eq!(info.client_id, 239346);
        assert_eq!(info.pdev, "0000:0f:00.0");
        assert_eq!(info.vram_kb, 84436);
        assert_eq!(info.gtt_kb, 26568); // key has a space before the tab
        assert_eq!(info.engines["gfx"], 809_494_162);
        assert_eq!(info.engines["enc"], 1_062_490_355);
        assert_eq!(info.engines.len(), 2);
    }

    #[test]
    fn fdinfo_of_a_regular_file_is_not_a_client() {
        assert_eq!(
            parse_fdinfo("pos:\t0\nflags:\t02\nmnt_id:\t3\nino:\t9\n"),
            None
        );
        // a driver line alone is not enough to tell clients apart
        assert_eq!(parse_fdinfo("drm-driver:\tamdgpu\n"), None);
    }

    #[test]
    fn fdinfo_skips_capacity_and_falls_back_to_resident_memory() {
        let text = "drm-driver:\ti915\ndrm-client-id:\t7\ndrm-pdev:\t0000:00:02.0\n\
            drm-engine-render:\t500 ns\ndrm-engine-capacity-video:\t2\n\
            drm-engine-video:\tgarbage ns\ndrm-resident-vram:\t2 MiB\n";
        let info = parse_fdinfo(text).unwrap();
        assert_eq!(info.engines.len(), 1);
        assert_eq!(info.engines["render"], 500);
        assert_eq!(info.vram_kb, 2048);
    }

    #[test]
    fn sizes_are_normalised_to_kib() {
        assert_eq!(parse_kib("10 KiB"), Some(10));
        assert_eq!(parse_kib("3 MiB"), Some(3072));
        assert_eq!(parse_kib("1 GiB"), Some(1024 * 1024));
        assert_eq!(parse_kib("4096"), Some(4));
        assert_eq!(parse_kib("x KiB"), None);
    }

    struct FakeGpu {
        dir: TempDir,
    }

    impl FakeGpu {
        /// One amdgpu card (`card1` / `renderD128`) at 0000:0f:00.0.
        fn new(with_dev_node: bool) -> Self {
            let dir = TempDir::new("drm");
            let pci = "sys/devices/pci0000:00/0000:0f:00.0";
            dir.mkdir(&format!("{pci}/hwmon/hwmon2"));
            dir.mkdir("sys/bus/pci/drivers/amdgpu");
            dir.symlink("sys/bus/pci/drivers/amdgpu", &format!("{pci}/driver"));
            dir.write(&format!("{pci}/vendor"), "0x1002\n");
            dir.write(&format!("{pci}/device"), "0x15bf\n");
            dir.write(&format!("{pci}/power/runtime_status"), "active\n");
            dir.write(&format!("{pci}/gpu_busy_percent"), "40\n");
            dir.write(&format!("{pci}/mem_info_vram_used"), "536870912\n");
            dir.write(&format!("{pci}/mem_info_vram_total"), "2147483648\n");
            dir.write(&format!("{pci}/mem_info_gtt_used"), "1000\n");
            dir.write(&format!("{pci}/mem_info_gtt_total"), "2000\n");
            dir.write(&format!("{pci}/hwmon/hwmon2/temp1_input"), "53000\n");
            dir.write(&format!("{pci}/hwmon/hwmon2/power1_average"), "29160000\n");
            dir.write(&format!("{pci}/hwmon/hwmon2/freq1_input"), "800000000\n");
            for node in ["card1", "renderD128"] {
                dir.symlink(pci, &format!("sys/class/drm/{node}/device"));
            }
            // connectors and the version file must not be taken for cards
            dir.mkdir("sys/class/drm/card1-HDMI-A-1");
            dir.write("sys/class/drm/version", "drm 1.1.0\n");
            dir.mkdir("dev/dri");
            if with_dev_node {
                dir.write("dev/dri/renderD128", "");
            }
            dir.mkdir("proc");
            FakeGpu { dir }
        }

        fn pci(&self, file: &str) -> String {
            format!("sys/devices/pci0000:00/0000:0f:00.0/{file}")
        }

        fn sampler(&self) -> GpuSampler {
            GpuSampler::new(
                &self.dir.path("proc"),
                &self.dir.path("sys"),
                &self.dir.path("dev"),
            )
        }

        /// Gives `pid` a descriptor `fd` for DRM client `client`.
        fn client(&self, pid: i32, fd: u32, client: u64, engines: &[(&str, u64)]) {
            self.dir
                .symlink_raw("/dev/dri/renderD128", &format!("proc/{pid}/fd/{fd}"));
            let mut text = format!(
                "pos:\t0\ndrm-driver:\tamdgpu\ndrm-client-id:\t{client}\n\
                 drm-pdev:\t0000:0f:00.0\ndrm-memory-vram:\t1024 KiB\ndrm-memory-gtt: \t512 KiB\n"
            );
            for (name, ns) in engines {
                text.push_str(&format!("drm-engine-{name}:\t{ns} ns\n"));
            }
            self.dir.write(&format!("proc/{pid}/fdinfo/{fd}"), &text);
        }
    }

    #[test]
    fn discovers_cards_but_not_connectors() {
        let fake = FakeGpu::new(true);
        let cards = discover(&fake.dir.path("sys"), &fake.dir.path("dev"));
        assert_eq!(cards.len(), 1);
        let card = &cards[0];
        assert_eq!(card.card, "card1");
        assert_eq!(card.pdev, "0000:0f:00.0");
        assert_eq!(card.driver, "amdgpu");
        assert_eq!(card.vendor, "0x1002");
        assert_eq!(card.render_node.as_deref(), Some("renderD128"));
        assert!(card.node_present);
    }

    #[test]
    fn a_gpu_without_a_device_node_is_reported_as_absent() {
        let fake = FakeGpu::new(false);
        let cards = discover(&fake.dir.path("sys"), &fake.dir.path("dev"));
        assert!(!cards[0].node_present);
    }

    #[test]
    fn reads_device_sensors_and_memory() {
        let fake = FakeGpu::new(true);
        let mut sampler = fake.sampler();
        let usage = &sampler.sample(&[], 0.0)[0];
        assert_eq!(usage.busy, Some(40.0)); // no averaged readings yet
        assert_eq!(usage.busy_samples, 0);
        assert_eq!(usage.vram_used, Some(536_870_912));
        assert_eq!(usage.vram_total, Some(2_147_483_648));
        assert_eq!(usage.gtt_total, Some(2000));
        assert_eq!(usage.temp, Some(53.0));
        assert_eq!(usage.power, Some(29.16));
        assert_eq!(usage.freq_mhz, Some(800.0));
        assert_eq!(usage.runtime_status.as_deref(), Some("active"));
    }

    #[test]
    fn busy_is_the_mean_of_the_background_readings() {
        let fake = FakeGpu::new(true);
        let mut sampler = fake.sampler();
        for load in [10, 20, 60] {
            fake.dir
                .write(&fake.pci("gpu_busy_percent"), &format!("{load}\n"));
            sampler.averager().tick();
        }
        let usage = &sampler.sample(&[], 0.0)[0];
        assert_eq!(usage.busy, Some(30.0));
        assert_eq!(usage.busy_samples, 3);
        // the readings are consumed
        fake.dir.write(&fake.pci("gpu_busy_percent"), "5\n");
        assert_eq!(sampler.sample(&[], 1.0)[0].busy, Some(5.0));
    }

    #[test]
    fn a_suspended_gpu_is_not_read() {
        let fake = FakeGpu::new(true);
        fake.dir
            .write(&fake.pci("power/runtime_status"), "suspended\n");
        let mut sampler = fake.sampler();
        sampler.averager().tick();
        let usage = &sampler.sample(&[], 0.0)[0];
        assert_eq!(usage.busy, None);
        assert_eq!(usage.vram_used, None);
        assert_eq!(usage.temp, None);
        assert_eq!(usage.runtime_status.as_deref(), Some("suspended"));
    }

    #[test]
    fn engine_usage_is_the_delta_between_samples() {
        let fake = FakeGpu::new(true);
        fake.client(100, 4, 1, &[("gfx", 1_000_000_000), ("enc", 5_000_000_000)]);
        let mut sampler = fake.sampler();

        // no interval yet, so no usage, but memory is known
        let first = &sampler.sample(&[100], 0.0)[0];
        assert!(first.engines.is_empty());
        assert_eq!(first.clients[0].vram_kb, 1024);

        // over 10 s: 0.5 s of gfx, 2 s of enc
        fake.client(100, 4, 1, &[("gfx", 1_500_000_000), ("enc", 7_000_000_000)]);
        let second = &sampler.sample(&[100], 10.0)[0];
        assert!((second.engines["gfx"] - 5.0).abs() < 1e-9);
        assert!((second.engines["enc"] - 20.0).abs() < 1e-9);
        assert!((second.clients[0].engines["enc"] - 20.0).abs() < 1e-9);
        assert!(!second.has_decode_engine);
    }

    #[test]
    fn a_client_on_several_descriptors_and_processes_counts_once() {
        let fake = FakeGpu::new(true);
        let mut sampler = fake.sampler();
        fake.client(100, 4, 1, &[("gfx", 0)]);
        sampler.sample(&[100, 101], 0.0);

        // the same client on two descriptors, and inherited by a child
        fake.client(100, 4, 1, &[("gfx", 1_000_000_000)]);
        fake.client(100, 5, 1, &[("gfx", 1_000_000_000)]);
        fake.client(101, 4, 1, &[("gfx", 1_000_000_000)]);
        let usage = &sampler.sample(&[100, 101], 10.0)[0];
        assert!((usage.engines["gfx"] - 10.0).abs() < 1e-9);
        assert_eq!(usage.clients.len(), 1);
        assert_eq!(usage.clients[0].pid, 100);
        assert_eq!(usage.clients[0].vram_kb, 1024);
    }

    #[test]
    fn a_new_client_counts_from_zero_and_totals_are_capped() {
        let fake = FakeGpu::new(true);
        fake.client(100, 4, 1, &[("gfx", 50_000_000_000)]);
        let mut sampler = fake.sampler();
        sampler.sample(&[100, 200, 201], 0.0);

        // pid 200 opened the GPU during the interval: 1 s of 10 s
        fake.client(200, 4, 2, &[("gfx", 1_000_000_000), ("dec", 1)]);
        let usage = &sampler.sample(&[100, 200, 201], 10.0)[0];
        let new = usage.clients.iter().find(|c| c.pid == 200).unwrap();
        assert!((new.engines["gfx"] - 10.0).abs() < 1e-9);
        assert!(usage.has_decode_engine);

        // two processes at 80% of one engine cannot exceed the engine
        fake.client(200, 4, 2, &[("gfx", 9_000_000_000)]);
        fake.client(201, 4, 3, &[("gfx", 8_000_000_000)]);
        let usage = &sampler.sample(&[100, 200, 201], 20.0)[0];
        assert_eq!(usage.engines["gfx"], 100.0);
    }

    #[test]
    fn processes_that_are_not_sampled_or_gone_are_ignored() {
        let fake = FakeGpu::new(true);
        fake.client(100, 4, 1, &[("gfx", 1)]);
        let mut sampler = fake.sampler();
        // 999 does not exist; 100 is not in the list
        let usage = &sampler.sample(&[999], 0.0)[0];
        assert!(usage.clients.is_empty());
    }

    #[test]
    fn unsupported_drivers_are_left_to_their_own_collectors() {
        let fake = FakeGpu::new(true);
        let driver = fake.dir.path(&fake.pci("driver"));
        std::fs::remove_file(&driver).unwrap();
        fake.dir.mkdir("sys/bus/pci/drivers/i915");
        fake.dir
            .symlink("sys/bus/pci/drivers/i915", &fake.pci("driver"));
        let mut sampler = fake.sampler();
        assert!(!sampler.has_cards());
        assert!(sampler.sample(&[], 0.0).is_empty());
    }
}
