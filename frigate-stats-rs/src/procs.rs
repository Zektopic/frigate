//! Per-process and system CPU / memory usage from procfs.
//!
//! CPU usage is the share of one core a process used since the previous
//! sample, like `top` (it can exceed 100 for a multi-threaded process). The
//! previous counters are kept here, so the first sample of a process falls
//! back to its lifetime average instead of reporting zero.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

/// The fields of `/proc/<pid>/stat` the monitor uses.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcStat {
    pub pid: i32,
    pub comm: String,
    pub ppid: i32,
    /// utime + stime, in clock ticks.
    pub ticks: u64,
    /// Start time since boot, in clock ticks.
    pub start_ticks: u64,
    pub rss_pages: u64,
    pub threads: u32,
}

/// Parses one `/proc/<pid>/stat` line.
///
/// The command name is wrapped in parentheses and may itself contain spaces
/// and parentheses, so the fields are counted from the last `)`.
pub fn parse_stat(line: &str) -> Option<ProcStat> {
    let open = line.find('(')?;
    let close = line.rfind(')')?;
    if close < open {
        return None;
    }
    let pid = line[..open].trim().parse().ok()?;
    let comm = line[open + 1..close].to_string();
    let rest: Vec<&str> = line[close + 1..].split_ascii_whitespace().collect();
    // rest[0] is field 3 (state); field N is rest[N - 3].
    let field = |n: usize| rest.get(n - 3).copied();
    let utime: u64 = field(14)?.parse().ok()?;
    let stime: u64 = field(15)?.parse().ok()?;
    Some(ProcStat {
        pid,
        comm,
        ppid: field(4)?.parse().ok()?,
        ticks: utime + stime,
        start_ticks: field(22)?.parse().ok()?,
        // rss is signed in the kernel's format string; never negative in practice
        rss_pages: field(24)?.parse::<i64>().ok()?.max(0) as u64,
        threads: field(20)?.parse::<i64>().ok()?.max(0) as u32,
    })
}

/// Aggregate CPU time of the machine, in clock ticks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpuTimes {
    pub busy: u64,
    pub total: u64,
}

/// Parses the aggregate `cpu` line of `/proc/stat`.
///
/// Matches psutil: guest time is already included in user/nice, and idle and
/// iowait are the only states that do not count as busy.
pub fn parse_cpu_line(line: &str) -> Option<CpuTimes> {
    let mut parts = line.split_ascii_whitespace();
    if parts.next()? != "cpu" {
        return None;
    }
    let values: Vec<u64> = parts.filter_map(|v| v.parse().ok()).collect();
    if values.len() < 4 {
        return None;
    }
    // user nice system idle iowait irq softirq steal [guest guest_nice]
    let total: u64 = values.iter().take(8).sum();
    let idle = values[3] + values.get(4).copied().unwrap_or(0);
    Some(CpuTimes {
        busy: total.saturating_sub(idle),
        total,
    })
}

/// Returns `(MemTotal, MemAvailable)` in kB from `/proc/meminfo`.
pub fn parse_meminfo(text: &str) -> Option<(u64, u64)> {
    let mut total = None;
    let mut available = None;
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let kb = || value.split_ascii_whitespace().next()?.parse::<u64>().ok();
        match key {
            "MemTotal" => total = kb(),
            "MemAvailable" => available = kb(),
            _ => {}
        }
        if total.is_some() && available.is_some() {
            break;
        }
    }
    Some((total?, available?))
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SystemUsage {
    /// Busy share of all cores since the previous sample, 0-100.
    pub cpu: f64,
    /// Used share of physical memory, 0-100.
    pub mem: f64,
    pub mem_total_kb: u64,
    pub mem_available_kb: u64,
    /// The cgroup memory limit, when the container has one.
    pub mem_limit_kb: Option<u64>,
    pub cpus: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProcUsage {
    pub pid: i32,
    pub ppid: i32,
    pub comm: String,
    pub cmdline: String,
    pub threads: u32,
    /// Share of one core since the previous sample.
    pub cpu: f64,
    /// Share of one core since the process started.
    pub cpu_average: f64,
    pub rss_kb: u64,
    /// Resident memory as a share of the memory limit (or of physical memory).
    pub mem: f64,
    /// `cpu` of this process plus all of its descendants.
    pub tree_cpu: f64,
    /// `mem` of this process plus all of its descendants.
    pub tree_mem: f64,
    pub tree_rss_kb: u64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProcSample {
    pub system: SystemUsage,
    /// Sorted by pid.
    pub procs: Vec<ProcUsage>,
}

pub struct ProcSampler {
    proc_root: PathBuf,
    cgroup_root: PathBuf,
    clk_tck: f64,
    page_kb: f64,
    /// pid -> (start time, cpu ticks) at the previous sample. The start time
    /// tells a reused pid apart from the process that had it before.
    prev: HashMap<i32, (u64, u64)>,
    prev_at: Option<f64>,
    prev_cpu: Option<CpuTimes>,
}

impl ProcSampler {
    pub fn new(proc_root: &Path, cgroup_root: &Path, clk_tck: u64, page_size: u64) -> Self {
        ProcSampler {
            proc_root: proc_root.to_path_buf(),
            cgroup_root: cgroup_root.to_path_buf(),
            clk_tck: clk_tck.max(1) as f64,
            page_kb: page_size as f64 / 1024.0,
            prev: HashMap::new(),
            prev_at: None,
            prev_cpu: None,
        }
    }

    fn read(&self, rel: &str) -> Option<String> {
        fs::read_to_string(self.proc_root.join(rel)).ok()
    }

    /// The cgroup v2 memory limit in kB, if one is set.
    fn memory_limit_kb(&self) -> Option<u64> {
        let text = fs::read_to_string(self.cgroup_root.join("memory.max")).ok()?;
        let bytes: u64 = text.trim().parse().ok()?; // "max" means unlimited
        (bytes > 0).then_some(bytes / 1024)
    }

    fn read_all_stats(&self) -> Vec<ProcStat> {
        let Ok(entries) = fs::read_dir(&self.proc_root) else {
            return Vec::new();
        };
        entries
            .flatten()
            .filter_map(|e| {
                let name = e.file_name();
                let name = name.to_str()?;
                if !name.bytes().all(|b| b.is_ascii_digit()) {
                    return None;
                }
                parse_stat(&self.read(&format!("{name}/stat"))?)
            })
            .collect()
    }

    fn read_cmdline(&self, pid: i32) -> String {
        let Ok(raw) = fs::read(self.proc_root.join(format!("{pid}/cmdline"))) else {
            return String::new();
        };
        let text = String::from_utf8_lossy(&raw);
        let joined = text
            .split('\0')
            .filter(|arg| !arg.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        joined.trim_end().to_string()
    }

    /// Samples the process tree under `root_pid`, plus every process whose
    /// name contains one of `extra_names` (and their descendants), for
    /// helpers that are supervised outside of the main process.
    ///
    /// `now` is a monotonic clock reading in seconds.
    pub fn sample(&mut self, root_pid: i32, extra_names: &[String], now: f64) -> ProcSample {
        let dt = self.prev_at.map(|t| now - t).filter(|d| *d > 0.0);
        let uptime: f64 = self
            .read("uptime")
            .and_then(|s| s.split_ascii_whitespace().next()?.parse().ok())
            .unwrap_or(0.0);

        // system
        let stat_text = self.read("stat").unwrap_or_default();
        let cpu_now = stat_text.lines().next().and_then(parse_cpu_line);
        let cpus = stat_text
            .lines()
            .filter(|l| {
                l.strip_prefix("cpu")
                    .is_some_and(|r| r.starts_with(|c: char| c.is_ascii_digit()))
            })
            .count();
        let system_cpu = match (cpu_now, self.prev_cpu) {
            (Some(now), Some(prev)) if now.total > prev.total => {
                100.0 * now.busy.saturating_sub(prev.busy) as f64 / (now.total - prev.total) as f64
            }
            // no interval yet: busy share since boot
            (Some(now), None) if now.total > 0 => 100.0 * now.busy as f64 / now.total as f64,
            _ => 0.0,
        };
        self.prev_cpu = cpu_now.or(self.prev_cpu);

        let (mem_total_kb, mem_available_kb) = self
            .read("meminfo")
            .and_then(|t| parse_meminfo(&t))
            .unwrap_or((0, 0));
        let mem_limit_kb = self.memory_limit_kb();
        let system = SystemUsage {
            cpu: system_cpu.clamp(0.0, 100.0),
            mem: if mem_total_kb > 0 {
                100.0 * mem_total_kb.saturating_sub(mem_available_kb) as f64 / mem_total_kb as f64
            } else {
                0.0
            },
            mem_total_kb,
            mem_available_kb,
            mem_limit_kb,
            cpus,
        };

        // process table
        let stats = self.read_all_stats();
        let mut children: HashMap<i32, Vec<i32>> = HashMap::new();
        for s in &stats {
            children.entry(s.ppid).or_default().push(s.pid);
        }
        let by_pid: HashMap<i32, &ProcStat> = stats.iter().map(|s| (s.pid, s)).collect();

        let mut selected: HashSet<i32> = HashSet::new();
        let mut stack: Vec<i32> = Vec::new();
        if by_pid.contains_key(&root_pid) {
            stack.push(root_pid);
        }
        stack.extend(
            stats
                .iter()
                .filter(|s| {
                    extra_names
                        .iter()
                        .any(|n| !n.is_empty() && s.comm.contains(n.as_str()))
                })
                .map(|s| s.pid),
        );
        while let Some(pid) = stack.pop() {
            if selected.insert(pid) {
                if let Some(kids) = children.get(&pid) {
                    stack.extend(kids);
                }
            }
        }

        let mem_base_kb = mem_limit_kb.unwrap_or(mem_total_kb) as f64;
        let mut usages: BTreeMap<i32, ProcUsage> = BTreeMap::new();
        let mut next_prev = HashMap::with_capacity(selected.len());
        for &pid in &selected {
            let s = by_pid[&pid];
            next_prev.insert(pid, (s.start_ticks, s.ticks));

            let cmdline = self.read_cmdline(pid);
            if cmdline.is_empty() {
                // kernel thread or zombie
                continue;
            }

            let lifetime = uptime - s.start_ticks as f64 / self.clk_tck;
            let cpu_average = if lifetime > 0.0 {
                100.0 * s.ticks as f64 / self.clk_tck / lifetime
            } else {
                0.0
            };
            let cpu = match (self.prev.get(&pid), dt) {
                (Some(&(start, ticks)), Some(dt)) if start == s.start_ticks => {
                    100.0 * s.ticks.saturating_sub(ticks) as f64 / self.clk_tck / dt
                }
                // first sample of this process
                _ => cpu_average,
            };
            let rss_kb = (s.rss_pages as f64 * self.page_kb) as u64;
            let mem = if mem_base_kb > 0.0 {
                100.0 * rss_kb as f64 / mem_base_kb
            } else {
                0.0
            };
            usages.insert(
                pid,
                ProcUsage {
                    pid,
                    ppid: s.ppid,
                    comm: s.comm.clone(),
                    cmdline,
                    threads: s.threads,
                    cpu,
                    cpu_average,
                    rss_kb,
                    mem,
                    tree_cpu: cpu,
                    tree_mem: mem,
                    tree_rss_kb: rss_kb,
                },
            );
        }
        self.prev = next_prev;
        self.prev_at = Some(now);

        // Fold every process into each of its ancestors. A parent always has
        // a lower start time than its children, but pids wrap, so walk the
        // parent chain instead of relying on pid order.
        let own: Vec<(i32, f64, f64, u64)> = usages
            .values()
            .map(|u| (u.pid, u.cpu, u.mem, u.rss_kb))
            .collect();
        for (pid, cpu, mem, rss_kb) in own {
            let mut ancestor = by_pid.get(&pid).map(|s| s.ppid);
            let mut hops = 0;
            while let Some(a) = ancestor {
                if let Some(u) = usages.get_mut(&a) {
                    u.tree_cpu += cpu;
                    u.tree_mem += mem;
                    u.tree_rss_kb += rss_kb;
                }
                ancestor = by_pid.get(&a).map(|s| s.ppid).filter(|p| *p != a);
                hops += 1;
                if hops > 64 {
                    break; // malformed table; never loop forever
                }
            }
        }

        ProcSample {
            system,
            procs: usages.into_values().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    #[test]
    fn stat_with_spaces_and_parens_in_comm() {
        let line = "4242 (tmux: server (1)) S 1 4242 4242 0 -1 4194560 1402 1103 0 1 \
                    70 30 5 6 20 0 7 0 191969 4231168 702 18446744073709551615 0 0";
        let s = parse_stat(line).unwrap();
        assert_eq!(s.pid, 4242);
        assert_eq!(s.comm, "tmux: server (1)");
        assert_eq!(s.ppid, 1);
        assert_eq!(s.ticks, 100);
        assert_eq!(s.threads, 7);
        assert_eq!(s.start_ticks, 191969);
        assert_eq!(s.rss_pages, 702);
    }

    #[test]
    fn stat_rejects_truncated_lines() {
        assert_eq!(parse_stat(""), None);
        assert_eq!(parse_stat("12 (x) S 1 2 3"), None);
        assert_eq!(parse_stat("12 x S 1"), None);
    }

    #[test]
    fn cpu_line_counts_idle_and_iowait_as_idle() {
        let t = parse_cpu_line("cpu  100 10 50 800 40 0 0 0 7 3").unwrap();
        // guest columns are not added again
        assert_eq!(t.total, 1000);
        assert_eq!(t.busy, 160);
        assert_eq!(parse_cpu_line("cpu0 1 2 3 4"), None);
        assert_eq!(parse_cpu_line("intr 1 2 3 4"), None);
    }

    #[test]
    fn meminfo_reads_total_and_available() {
        let text = "MemTotal:       16000000 kB\nMemFree:  1 kB\nMemAvailable:    4000000 kB\n";
        assert_eq!(parse_meminfo(text), Some((16_000_000, 4_000_000)));
        assert_eq!(parse_meminfo("MemTotal: 5 kB\n"), None);
    }

    /// A fake procfs: `add(pid, ppid, comm, cmdline, ticks, start, rss_pages)`.
    struct FakeProc {
        dir: TempDir,
    }

    impl FakeProc {
        fn new() -> Self {
            let dir = TempDir::new("procs");
            let f = FakeProc { dir };
            f.set_uptime(1000.0);
            f.set_cpu(0, 0);
            f.dir.write(
                "proc/meminfo",
                "MemTotal: 1000000 kB\nMemAvailable: 250000 kB\n",
            );
            f
        }

        fn set_uptime(&self, secs: f64) {
            self.dir.write("proc/uptime", &format!("{secs:.2} 0.00\n"));
        }

        fn set_cpu(&self, busy: u64, idle: u64) {
            self.dir.write(
                "proc/stat",
                &format!("cpu  {busy} 0 0 {idle} 0 0 0 0 0 0\ncpu0 1 0 0 1\ncpu1 1 0 0 1\n"),
            );
        }

        #[allow(clippy::too_many_arguments)]
        fn add(
            &self,
            pid: i32,
            ppid: i32,
            comm: &str,
            cmdline: &str,
            ticks: u64,
            start: u64,
            rss: u64,
        ) {
            self.dir.write(
                &format!("proc/{pid}/stat"),
                &format!(
                    "{pid} ({comm}) S {ppid} 0 0 0 -1 0 0 0 0 0 {ticks} 0 0 0 20 0 3 0 {start} 0 {rss} 0\n"
                ),
            );
            self.dir
                .write(&format!("proc/{pid}/cmdline"), &cmdline.replace(' ', "\0"));
        }

        fn sampler(&self) -> ProcSampler {
            // 100 ticks per second, 4 KiB pages
            ProcSampler::new(&self.dir.path("proc"), &self.dir.path("cgroup"), 100, 4096)
        }
    }

    fn find(sample: &ProcSample, pid: i32) -> &ProcUsage {
        sample.procs.iter().find(|p| p.pid == pid).unwrap()
    }

    #[test]
    fn cpu_is_the_delta_between_samples() {
        let fake = FakeProc::new();
        fake.add(10, 1, "python3", "python3 -u -m frigate", 5000, 0, 100);
        let mut sampler = fake.sampler();

        // first sample: lifetime average, 50 s of cpu over 1000 s
        let first = sampler.sample(10, &[], 0.0);
        assert!((find(&first, 10).cpu - 5.0).abs() < 1e-9);
        assert!((find(&first, 10).cpu_average - 5.0).abs() < 1e-9);

        // 3 s of cpu during the next 10 s -> 30%
        fake.add(10, 1, "python3", "python3 -u -m frigate", 5300, 0, 100);
        fake.set_uptime(1010.0);
        let second = sampler.sample(10, &[], 10.0);
        assert!((find(&second, 10).cpu - 30.0).abs() < 1e-9);
        // 53 s over 1010 s
        assert!((find(&second, 10).cpu_average - 100.0 * 53.0 / 1010.0).abs() < 1e-9);

        // idle interval
        fake.set_uptime(1020.0);
        let third = sampler.sample(10, &[], 20.0);
        assert_eq!(find(&third, 10).cpu, 0.0);
    }

    #[test]
    fn multi_threaded_process_can_exceed_100() {
        let fake = FakeProc::new();
        fake.add(10, 1, "ffmpeg", "ffmpeg -i x", 0, 0, 1);
        let mut sampler = fake.sampler();
        sampler.sample(10, &[], 0.0);
        fake.add(10, 1, "ffmpeg", "ffmpeg -i x", 2500, 0, 1);
        let s = sampler.sample(10, &[], 10.0);
        assert!((find(&s, 10).cpu - 250.0).abs() < 1e-9);
    }

    #[test]
    fn reused_pid_is_not_compared_with_the_old_process() {
        let fake = FakeProc::new();
        fake.add(10, 1, "python3", "python3 main", 10, 0, 1);
        fake.add(20, 10, "ffmpeg", "ffmpeg old", 90_000, 0, 1);
        let mut sampler = fake.sampler();
        sampler.sample(10, &[], 0.0);

        // pid 20 is now a different process, started 5 s ago, 1 s of cpu
        fake.set_uptime(1010.0);
        fake.add(20, 10, "ffmpeg", "ffmpeg new", 100, 100_500, 1);
        let s = sampler.sample(10, &[], 10.0);
        assert!((find(&s, 20).cpu - 20.0).abs() < 1e-9);
    }

    #[test]
    fn selects_the_tree_and_named_extras_only() {
        let fake = FakeProc::new();
        fake.add(1, 0, "s6-svscan", "s6-svscan /run", 1, 0, 1);
        fake.add(10, 1, "python3", "python3 -u -m frigate", 1, 0, 1);
        fake.add(11, 10, "python3", "frigate.detector:ncnn", 1, 0, 1);
        fake.add(
            12,
            11,
            "frigate-detecto",
            "/opt/frigate/frigate-detector-rs m.param",
            1,
            0,
            1,
        );
        fake.add(
            30,
            1,
            "go2rtc",
            "/usr/local/go2rtc/bin/go2rtc -config=x",
            1,
            0,
            1,
        );
        fake.add(31, 30, "ffmpeg", "ffmpeg -i rtsp://cam", 1, 0, 1);
        fake.add(40, 1, "nginx", "nginx: master process", 1, 0, 1);
        fake.add(50, 2, "kworker/0:1", "", 1, 0, 0);
        let mut sampler = fake.sampler();
        let s = sampler.sample(10, &["go2rtc".into(), "certsync".into()], 0.0);
        let pids: Vec<i32> = s.procs.iter().map(|p| p.pid).collect();
        assert_eq!(pids, vec![10, 11, 12, 30, 31]);
        assert_eq!(
            find(&s, 12).cmdline,
            "/opt/frigate/frigate-detector-rs m.param"
        );
    }

    #[test]
    fn tree_totals_include_all_descendants() {
        let fake = FakeProc::new();
        fake.add(10, 1, "python3", "python3 main", 0, 0, 100);
        fake.add(11, 10, "python3", "frigate.detector", 0, 0, 200);
        fake.add(12, 11, "frigate-detecto", "frigate-detector-rs", 0, 0, 300);
        let mut sampler = fake.sampler();
        sampler.sample(10, &[], 0.0);

        fake.add(10, 1, "python3", "python3 main", 100, 0, 100);
        fake.add(11, 10, "python3", "frigate.detector", 200, 0, 200);
        fake.add(
            12,
            11,
            "frigate-detecto",
            "frigate-detector-rs",
            800,
            0,
            300,
        );
        let s = sampler.sample(10, &[], 10.0);

        let detector = find(&s, 11);
        assert!((detector.cpu - 20.0).abs() < 1e-9);
        assert!((detector.tree_cpu - 100.0).abs() < 1e-9);
        assert_eq!(detector.tree_rss_kb, (200 + 300) * 4);
        let main = find(&s, 10);
        assert!((main.tree_cpu - 110.0).abs() < 1e-9);
        assert_eq!(main.tree_rss_kb, 600 * 4);
        // a leaf's tree total is its own usage
        assert_eq!(find(&s, 12).tree_cpu, find(&s, 12).cpu);
    }

    #[test]
    fn memory_is_relative_to_the_cgroup_limit_when_set() {
        let fake = FakeProc::new();
        // 25000 pages * 4 KiB = 100000 kB
        fake.add(10, 1, "python3", "python3 main", 0, 0, 25_000);
        let mut sampler = fake.sampler();
        let s = sampler.sample(10, &[], 0.0);
        assert_eq!(s.system.mem_limit_kb, None);
        assert!((find(&s, 10).mem - 10.0).abs() < 1e-9); // of 1000000 kB

        fake.dir.write("cgroup/memory.max", "max\n");
        assert_eq!(sampler.sample(10, &[], 1.0).system.mem_limit_kb, None);

        fake.dir.write("cgroup/memory.max", "409600000\n"); // 400000 kB
        let s = sampler.sample(10, &[], 2.0);
        assert_eq!(s.system.mem_limit_kb, Some(400_000));
        assert!((find(&s, 10).mem - 25.0).abs() < 1e-9);
    }

    #[test]
    fn system_cpu_and_memory() {
        let fake = FakeProc::new();
        fake.add(10, 1, "python3", "python3 main", 0, 0, 1);
        fake.set_cpu(100, 900);
        let mut sampler = fake.sampler();
        let first = sampler.sample(10, &[], 0.0);
        assert!((first.system.cpu - 10.0).abs() < 1e-9); // since boot
        assert!((first.system.mem - 75.0).abs() < 1e-9);
        assert_eq!(first.system.cpus, 2);

        fake.set_cpu(400, 1600); // +300 busy of +1000 total
        let second = sampler.sample(10, &[], 10.0);
        assert!((second.system.cpu - 30.0).abs() < 1e-9);
    }

    #[test]
    fn missing_root_and_empty_procfs_do_not_panic() {
        let dir = TempDir::new("procs-empty");
        let mut sampler = ProcSampler::new(&dir.path("proc"), &dir.path("cgroup"), 100, 4096);
        let s = sampler.sample(1234, &["go2rtc".into()], 0.0);
        assert!(s.procs.is_empty());
        assert_eq!(s.system.cpu, 0.0);
    }
}
