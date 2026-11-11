//! Throwaway directory trees standing in for /proc, /sys and /dev in tests.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

static NEXT: AtomicU32 = AtomicU32::new(0);

pub struct TempDir {
    root: PathBuf,
}

impl TempDir {
    pub fn new(label: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "frigate-stats-rs-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        TempDir { root }
    }

    pub fn path(&self, rel: &str) -> PathBuf {
        self.root.join(rel)
    }

    /// Writes a file, creating its parent directories.
    pub fn write(&self, rel: &str, content: &str) {
        let path = self.path(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }

    pub fn mkdir(&self, rel: &str) {
        fs::create_dir_all(self.path(rel)).unwrap();
    }

    /// Creates `link` pointing at `target` (both relative to the root).
    pub fn symlink(&self, target: &str, link: &str) {
        let link = self.path(link);
        fs::create_dir_all(link.parent().unwrap()).unwrap();
        let _ = fs::remove_file(&link);
        std::os::unix::fs::symlink(self.path(target), link).unwrap();
    }

    /// Creates `link` pointing at an arbitrary, possibly dangling, path.
    pub fn symlink_raw(&self, target: &str, link: &str) {
        let link = self.path(link);
        fs::create_dir_all(link.parent().unwrap()).unwrap();
        let _ = fs::remove_file(&link);
        std::os::unix::fs::symlink(target, link).unwrap();
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
