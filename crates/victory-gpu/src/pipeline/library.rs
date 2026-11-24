//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used.
//! 2. The title pipelines a run drew or dispatched with (made in time or
//!    not) are listed by a key known when the title creates them (their
//!    shader containers and graphics state); the next run hands those to
//!    the workers at creation, during loading. A full list keeps the
//!    latest run's first.
//!
//! 3. The pre-build (`victory translate --cache --pipelines`) lists the
//!    title pipelines it has put in the library, by the same key, so a
//!    second pre-build skips them without translating or loading anything.
//!    The list holds only for the library bytes it was saved with and for
//!    what names the pipelines (the translator, this module and `host`).
//!
//! The host's library is thread-safe and synchronises itself, except that
//! loads of one pipeline must not overlap (CreatePipelineLibrary, "Thread
//! Safety"): loads and stores take a lock for their name only, so workers
//! load in parallel and a save does not stop them. One lock over every
//! call made loads (which take up to seconds each) run one at a time, and
//! workers waited seconds for it.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, RwLock};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// Host pipelines created, loaded from the library, and title pipelines
/// started at creation, since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY: AtomicU64 = AtomicU64::new(0);

struct Library {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library.
    _blob: Vec<u8>,
    /// Pipelines stored since the last save.
    added: AtomicU64,
    /// Title pipelines the pre-build put in the library or found there, by
    /// key, and whether that list changed since the last save.
    prebuilt: Mutex<HashSet<String>>,
    prebuilt_changed: AtomicBool,
}

// SAFETY: the library is free-threaded (calls that must not overlap take a
// lock for their name); the blob is only read.
unsafe impl Send for Library {}
// SAFETY: as above.
unsafe impl Sync for Library {}

const LIBRARY_FILE: &str = "host-pipelines.bin";
const EARLY_FILE: &str = "early-pipelines.txt";
const PREBUILT_FILE: &str = "prebuilt-pipelines.txt";
/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 8192;

fn library() -> Option<&'static Library> {
    static LIBRARY: OnceLock<Option<Library>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let read = |file| DIR.get().and_then(|d| std::fs::read(d.join(file)).ok());
            let blob = read(LIBRARY_FILE).unwrap_or_default();
            let tag = tag(&blob);
            let mut library = open(&device, blob);
            // The pre-build's list, if it was saved with these bytes.
            if let (Some(l), Some(list)) = (library.as_mut(), read(PREBUILT_FILE))
                && !l._blob.is_empty()
            {
                l.prebuilt = Mutex::new(listed(&list, &tag));
            }
            if library.is_none() {
                tracing::warn!("no host pipeline library: pipelines are not kept");
            }
            library
        })
        .as_ref()
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
fn open(device: &ID3D12Device1, blob: Vec<u8>) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // SAFETY: the bytes outlive the library (kept beside it).
        unsafe {
            device.CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?.cast()
        }
    };
    if !blob.is_empty() {
        let started = std::time::Instant::now();
        match make(&blob) {
            Ok(library) => {
                let ms = started.elapsed().as_millis() as u64;
                tracing::info!(bytes = blob.len(), ms, "host pipeline library loaded");
                return Some(Library {
                    library,
                    _blob: blob,
                    added: AtomicU64::new(0),
                    prebuilt: Mutex::new(HashSet::new()),
                    prebuilt_changed: AtomicBool::new(false),
                });
            }
            Err(e) => tracing::info!(error = %e, "host pipeline library rejected; starting an empty one"),
        }
    } else {
        tracing::info!("no host pipeline library yet; starting an empty one");
    }
    let library = make(&[]).ok()?;
    Some(Library {
        library,
        _blob: Vec::new(),
        added: AtomicU64::new(0),
        prebuilt: Mutex::new(HashSet::new()),
        prebuilt_changed: AtomicBool::new(false),
    })
}

/// What the pre-build's list holds for: the library's bytes, and what
/// names its pipelines (the translation, as the shader cache keys it; the
/// host descriptions).
fn tag(blob: &[u8]) -> String {
    let limit = victory_shader::emit::loop_limit().to_le_bytes();
    let translator = victory_shader::SOURCE_HASH.as_bytes();
    let mut h = blake3::Hasher::new();
    for part in [
        blob,
        translator,
        include_bytes!("../shader_cache.rs"),
        &limit,
        include_bytes!("host.rs"),
        include_bytes!("library.rs"),
    ] {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex().to_string()
}

/// The pre-build's list as saved: the tag it holds for, then its keys.
fn list(tag: &str, keys: &HashSet<String>) -> String {
    let keys = keys.iter().map(String::as_str);
    std::iter::once(tag).chain(keys).collect::<Vec<_>>().join("\n")
}

/// The keys of a saved list, if it holds for `tag` (else none).
fn listed(list: &[u8], tag: &str) -> HashSet<String> {
    let list = String::from_utf8_lossy(list);
    let mut lines = list.lines();
    match lines.next() == Some(tag) {
        true => lines.map(str::to_owned).collect(),
        false => HashSet::new(),
    }
}

/// Whether the pre-build has the title pipeline `key` in the library.
pub(crate) fn prebuilt(key: &str) -> bool {
    library().is_some_and(|l| l.prebuilt.lock().unwrap_or_else(|e| e.into_inner()).contains(key))
}

/// The pre-build has the title pipeline `key` in the library now.
pub(crate) fn note_prebuilt(key: String) {
    if let Some(l) = library() {
        if l.prebuilt.lock().unwrap_or_else(|e| e.into_inner()).insert(key) {
            l.prebuilt_changed.store(true, Ordering::Relaxed);
        }
    }
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root].into_iter().chain(code.iter().copied()).chain([state.as_bytes()]) {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex()[..32].to_owned()
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored.
pub(crate) fn pipeline(
    name: &str,
    load: impl FnOnce(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let wide = HSTRING::from(name);
    #[cfg(feature = "diagnostics")]
    LAST.set(Some((name.to_owned(), true)));
    let Some(library) = library() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    let loaded = {
        let _name = lock_name(name);
        load(&library.library, PCWSTR(wide.as_ptr()))
    };
    if let Ok(pso) = loaded {
        LOADED.fetch_add(1, Ordering::Relaxed);
        return Ok(pso);
    }
    // Compiled outside the locks: compiles take up to seconds.
    #[cfg(feature = "diagnostics")]
    LAST.set(Some((name.to_owned(), false)));
    let pso = create()?;
    MADE.fetch_add(1, Ordering::Relaxed);
    let _name = lock_name(name);
    let _store = STORES.read().unwrap_or_else(|e| e.into_inner());
    // SAFETY: a live pipeline; a name already stored (a description that
    // differs) is refused, which leaves the library as it was.
    if unsafe { library.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
        library.added.fetch_add(1, Ordering::Relaxed);
    }
    Ok(pso)
}

/// Held shared by stores and alone by a save: the serialised size must not
/// change between asking for it and serialising. Loads do not change it.
static STORES: RwLock<()> = RwLock::new(());

/// The lock for loads and stores of pipelines named `name` (a hex hash):
/// one of a few, so that pipelines of one name never load at once.
fn lock_name(name: &str) -> MutexGuard<'static, ()> {
    static NAMES: [Mutex<()>; 64] = [const { Mutex::new(()) }; 64];
    let stripe = name.bytes().fold(0usize, |h, b| h.wrapping_mul(31).wrapping_add(b.into())) % NAMES.len();
    NAMES[stripe].lock().unwrap_or_else(|e| e.into_inner())
}

/// The first wait between periodic saves; it doubles up to `SAVE_MAX`.
const SAVE_EVERY: Duration = Duration::from_secs(20);
const SAVE_MAX: Duration = Duration::from_secs(160);
/// A periodic save waits at least this many times as long as the last took
/// (the library grows past a gigabyte; serialising it holds back stores,
/// and writing it takes seconds of disk).
const COST_FACTOR: u32 = 30;

/// The wait before the next periodic save, after `saves` of them, the last
/// of which took `cost`.
fn save_interval(saves: u32, cost: Duration) -> Duration {
    let backoff = SAVE_EVERY.saturating_mul(1 << saves.min(16)).min(SAVE_MAX);
    backoff.max(cost.saturating_mul(COST_FACTOR).min(Duration::from_secs(600)))
}

struct Schedule {
    /// When a periodic save may run next (`None`: at once).
    next: Option<Instant>,
    saves: u32,
}

#[cfg(feature = "diagnostics")]
thread_local! {
    /// This thread's last pipeline: its name, and whether it was loaded
    /// (else made), for the pipeline log (diagnostics).
    pub(crate) static LAST: std::cell::Cell<Option<(String, bool)>> =
        const { std::cell::Cell::new(None) };
}

/// Writes the library and the early list if pipelines were added since the
/// last save and the periodic wait has passed (or `now`, at the end of a
/// run). A periodic save while another is running is skipped: workers call
/// this after every pipeline and must not wait for a save.
pub(crate) fn save(now: bool) {
    // Held over serialisation and both writes, so overlapping saves cannot
    // replace files under each other.
    static SCHEDULE: Mutex<Schedule> = Mutex::new(Schedule { next: None, saves: 0 });
    let (Some(dir), Some(library)) = (DIR.get(), library()) else { return };
    let mut schedule = match SCHEDULE.try_lock() {
        Ok(s) => s,
        Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) if now => SCHEDULE.lock().unwrap_or_else(|e| e.into_inner()),
        Err(std::sync::TryLockError::WouldBlock) => return,
    };
    if !now && schedule.next.is_some_and(|t| Instant::now() < t) {
        return;
    }
    let started = Instant::now();
    let wait = match save_files(dir, library) {
        Some(()) => {
            schedule.saves += 1;
            save_interval(schedule.saves - 1, started.elapsed())
        }
        None => SAVE_EVERY,
    };
    schedule.next = Some(Instant::now() + wait);
}

/// Writes the library and the early list if anything was added; `Some`
/// if they were written.
fn save_files(dir: &std::path::Path, library: &Library) -> Option<()> {
    let (blob, prebuilt, saved_additions) = {
        let _stores = STORES.write().unwrap_or_else(|e| e.into_inner());
        let added = library.added.load(Ordering::Relaxed);
        if added == 0 && !library.prebuilt_changed.load(Ordering::Relaxed) && !USED_CHANGED.swap(false, Ordering::Relaxed) {
            return None;
        }
        // SAFETY: a buffer of the size the library asks for, which no
        // store changes meanwhile.
        let bytes = unsafe {
            let mut bytes = vec![0u8; library.library.GetSerializedSize()];
            library.library.Serialize(&mut bytes).map(|()| bytes)
        };
        match bytes {
            Ok(bytes) => {
                library.prebuilt_changed.store(false, Ordering::Relaxed);
                // Kept beside the bytes it holds for (also when a run adds
                // pipelines).
                let keys = library.prebuilt.lock().unwrap_or_else(|e| e.into_inner());
                let prebuilt = (!keys.is_empty()).then(|| list(&tag(&bytes), &keys));
                (bytes, prebuilt, added)
            }
            Err(e) => {
                tracing::warn!(error = %e, "host pipeline library not serialised");
                return None;
            }
        }
    };
    // Pipelines stored while the files are written count for the next save.
    library.added.fetch_sub(saved_additions, Ordering::Relaxed);
    let used = USED.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(Used::list).unwrap_or_default();
    let written = std::fs::create_dir_all(dir)
        .and_then(|()| write_atomically(&dir.join(LIBRARY_FILE), &blob))
        .and_then(|()| write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()))
        .and_then(|()| match &prebuilt {
            Some(list) => write_atomically(&dir.join(PREBUILT_FILE), list.as_bytes()),
            None => Ok(()),
        });
    let (created, distinct) = created();
    let (shaders_cached, shaders_translated) = crate::shader_cache::counts();
    match written {
        Ok(()) => {
            tracing::info!(
                bytes = blob.len(),
                early = used.len(),
                made = MADE.load(Ordering::Relaxed),
                loaded = LOADED.load(Ordering::Relaxed),
                started_early = EARLY.load(Ordering::Relaxed),
                created,
                distinct,
                shaders_cached,
                shaders_translated,
                "host pipeline library saved"
            );
            Some(())
        }
        Err(e) => {
            tracing::warn!(error = %e, "host pipeline library not saved");
            None
        }
    }
}

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

/// Title pipelines created, and their distinct keys.
static CREATED: Mutex<(u64, Option<HashSet<String>>)> = Mutex::new((0, None));

/// The title created a pipeline with `key`.
pub(crate) fn note_created(key: &str) {
    let mut created = CREATED.lock().unwrap_or_else(|e| e.into_inner());
    created.0 += 1;
    created.1.get_or_insert_with(HashSet::new).insert(key.to_owned());
}

/// Title pipelines created so far, and how many distinct ones.
fn created() -> (u64, usize) {
    let created = CREATED.lock().unwrap_or_else(|e| e.into_inner());
    (created.0, created.1.as_ref().map_or(0, HashSet::len))
}

/// Title pipeline keys drawn or dispatched with: in earlier runs (the
/// list as loaded), and in this one.
#[derive(Default)]
struct Used {
    before: HashSet<String>,
    now: HashSet<String>,
}

impl Used {
    /// The list to keep: this run's keys, then earlier runs' others, at
    /// most `EARLY_MAX` (the oldest go once it is full).
    fn list(&self) -> Vec<String> {
        let earlier = self.before.iter().filter(|k| !self.now.contains(*k));
        self.now.iter().chain(earlier).take(EARLY_MAX).cloned().collect()
    }
}

static USED: Mutex<Option<Used>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn used() -> std::sync::MutexGuard<'static, Option<Used>> {
    let mut used = USED.lock().unwrap_or_else(|e| e.into_inner());
    if used.is_none() {
        let listed = DIR
            .get()
            .and_then(|d| std::fs::read_to_string(d.join(EARLY_FILE)).ok())
            .unwrap_or_default();
        let before: HashSet<String> = listed.lines().filter(|l| !l.is_empty()).map(str::to_owned).collect();
        tracing::info!(pipelines = before.len(), "title pipelines to start early");
        *used = Some(Used { before, now: HashSet::new() });
    }
    used
}

/// Reads the early list now if not yet, so a save keeps it (a pre-build
/// asks for no title pipeline).
pub(crate) fn keep_early_list() {
    drop(used());
}

/// Whether the title pipeline with `key` was drawn with in an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    used().as_ref().is_some_and(|u| u.before.contains(key))
}

/// The title pipeline with `key` was drawn or dispatched with.
pub(crate) fn note_used(key: &str) {
    if used().get_or_insert_with(Used::default).now.insert(key.to_owned()) {
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    /// This run's keys stay in the list; earlier runs' fill it up to the cap.
    #[test]
    fn the_early_list_keeps_this_run_first() {
        let keys = |r: std::ops::Range<usize>| r.map(|i| format!("{i:032x}")).collect::<HashSet<_>>();
        let used = Used { before: keys(0..EARLY_MAX), now: keys(EARLY_MAX - 10..EARLY_MAX + 10) };
        let list = used.list();
        assert_eq!(list.len(), EARLY_MAX);
        assert!(used.now.iter().all(|k| list.contains(k)));
        assert_eq!(list.iter().collect::<HashSet<_>>().len(), EARLY_MAX, "no key twice");
    }

    /// The pre-build's list holds only for the library bytes it was
    /// saved with.
    #[test]
    fn the_prebuilt_list_holds_for_its_library_only() {
        let keys: HashSet<String> = ["a".repeat(32), "b".repeat(32)].into();
        let saved = list(&tag(b"library"), &keys);
        assert_eq!(listed(saved.as_bytes(), &tag(b"library")), keys);
        assert!(listed(saved.as_bytes(), &tag(b"library2")).is_empty());
        assert!(listed(b"", &tag(b"library")).is_empty());
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
            return; // no host GPU here
        };
        let library = open(&device, b"not a pipeline library".repeat(16)).expect("an empty library");
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(library._blob.is_empty());
    }

    #[test]
    fn periodic_saves_back_off_and_follow_their_cost() {
        let cheap = Duration::from_millis(100);
        let waits: Vec<u64> = (0..6).map(|n| save_interval(n, cheap).as_secs()).collect();
        assert_eq!(waits, [20, 40, 80, 160, 160, 160]);
        assert_eq!(save_interval(u32::MAX, cheap), SAVE_MAX);
        // A save of 4 s waits 2 minutes even at first.
        assert_eq!(save_interval(0, Duration::from_secs(4)), Duration::from_secs(120));
    }

    /// A name's lock is held while another name (another stripe) loads.
    #[test]
    fn loads_of_other_names_do_not_wait_for_a_held_name() {
        let held = lock_name("00");
        let other = std::thread::scope(|s| {
            s.spawn(|| {
                let _b = lock_name("01");
            })
            .join()
        });
        assert!(other.is_ok());
        drop(held);
    }
}
