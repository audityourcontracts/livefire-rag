use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use sha2::{Digest as _, Sha256};

use crate::{EmbeddingError, Result, validate_vector};

pub const EMBEDDING_SHARD_MAGIC: [u8; 8] = *b"LFREMB01";
pub const EMBEDDING_SHARD_HEADER_BYTES: u32 = 64;
const EMBEDDING_SHARD_VERSION: u16 = 1;
const EMBEDDING_SHARD_DTYPE_F32_LE: u8 = 1;
const EMBEDDING_SHARD_FLAGS: u8 = 0;
const EMBEDDING_SHARD_RESERVED: u32 = 0;

static STAGING_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingShardMetadata {
    pub row_count: u64,
    pub dimensions: u32,
    pub order_sha256: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingShardExpectation {
    pub row_count: u64,
    pub dimensions: u32,
    pub order_sha256: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedEmbeddingTaskPart {
    pub metadata: EmbeddingShardMetadata,
    pub bytes: u64,
    pub sha256: [u8; 32],
}

/// State returned before a higher layer decides whether a task needs to run.
/// A structurally valid orphan is reported as verified, but is not proof of a
/// completed task; durable receipt validation remains the caller's job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbeddingTaskPartPreparation {
    Missing,
    Verified {
        part: VerifiedEmbeddingTaskPart,
        /// A replacement was published after an earlier invalid part was
        /// quarantined. Call `complete_embedding_task_part_recovery` only
        /// after the higher layer has published its bound receipt.
        recovery_pending: bool,
    },
    Quarantined {
        quarantine_path: PathBuf,
    },
}

impl From<EmbeddingShardExpectation> for EmbeddingShardMetadata {
    fn from(value: EmbeddingShardExpectation) -> Self {
        Self {
            row_count: value.row_count,
            dimensions: value.dimensions,
            order_sha256: value.order_sha256,
        }
    }
}

impl From<EmbeddingShardMetadata> for EmbeddingShardExpectation {
    fn from(value: EmbeddingShardMetadata) -> Self {
        Self {
            row_count: value.row_count,
            dimensions: value.dimensions,
            order_sha256: value.order_sha256,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicPublishOutcome {
    Published,
    AlreadyExists,
}

/// A same-directory staging path that is made visible without overwriting an
/// existing result. Dropping an uncommitted publication removes its stage.
#[derive(Debug)]
pub struct AtomicFilePublication {
    destination: PathBuf,
    staging: PathBuf,
    committed: bool,
}

impl AtomicFilePublication {
    pub fn new(destination: &Path) -> Result<Self> {
        let parent = destination
            .parent()
            .ok_or(EmbeddingError::Invalid("embedding result parent"))?;
        fs::create_dir_all(parent)?;
        let file_name = destination
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or(EmbeddingError::Invalid("embedding result file name"))?;
        for _ in 0..128 {
            let sequence = STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let staging = parent.join(format!(
                ".{file_name}.{}.{}.partial",
                std::process::id(),
                sequence
            ));
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&staging)
            {
                Ok(file) => {
                    drop(file);
                    return Ok(Self {
                        destination: destination.to_owned(),
                        staging,
                        committed: false,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error.into()),
            }
        }
        Err(EmbeddingError::Invalid("embedding staging path exhaustion"))
    }

    #[must_use]
    pub fn staging_path(&self) -> &Path {
        &self.staging
    }

    #[must_use]
    pub fn destination(&self) -> &Path {
        &self.destination
    }

    pub fn commit(mut self) -> Result<AtomicPublishOutcome> {
        OpenOptions::new()
            .read(true)
            .open(&self.staging)?
            .sync_all()?;
        let outcome = match fs::hard_link(&self.staging, &self.destination) {
            Ok(()) => AtomicPublishOutcome::Published,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                AtomicPublishOutcome::AlreadyExists
            }
            Err(error) => return Err(error.into()),
        };
        fs::remove_file(&self.staging)?;
        self.committed = true;
        if let Some(parent) = self.destination.parent() {
            File::open(parent)?.sync_all()?;
        }
        Ok(outcome)
    }
}

impl Drop for AtomicFilePublication {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_file(&self.staging);
        }
    }
}

pub struct EmbeddingShardWriter {
    writer: BufWriter<File>,
    metadata: EmbeddingShardMetadata,
    rows_written: u64,
}

impl EmbeddingShardWriter {
    pub fn create(path: &Path, metadata: EmbeddingShardMetadata) -> Result<Self> {
        if metadata.dimensions == 0 {
            return Err(EmbeddingError::Invalid("embedding shard dimensions"));
        }
        let file = OpenOptions::new().write(true).truncate(true).open(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&encode_header(metadata))?;
        Ok(Self {
            writer,
            metadata,
            rows_written: 0,
        })
    }

    pub fn write_vector(&mut self, vector: &[f32]) -> Result<()> {
        let dimensions = usize::try_from(self.metadata.dimensions)
            .map_err(|_| EmbeddingError::Invalid("embedding shard dimensions"))?;
        if self.rows_written >= self.metadata.row_count || vector.len() != dimensions {
            return Err(EmbeddingError::Invalid("embedding shard vector shape"));
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(EmbeddingError::Invalid("embedding shard non-finite vector"));
        }
        for value in vector {
            self.writer.write_all(&value.to_le_bytes())?;
        }
        self.rows_written += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        if self.rows_written != self.metadata.row_count {
            return Err(EmbeddingError::Invalid("embedding shard row count"));
        }
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingShard {
    path: PathBuf,
    metadata: EmbeddingShardMetadata,
}

impl EmbeddingShard {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let metadata = read_header(&mut file)?;
        validate_file_length(&file, metadata)?;
        let shard = Self {
            path: path.to_owned(),
            metadata,
        };
        // A result is complete only if every scalar is a finite f32. Perform
        // that validation once when opening so corrupt results are never
        // treated as restart-safe completed work.
        for vector in shard.vectors()? {
            let _ = vector?;
        }
        Ok(shard)
    }

    pub fn open_expected(path: &Path, expected: EmbeddingShardExpectation) -> Result<Self> {
        let shard = Self::open(path)?;
        if shard.metadata != EmbeddingShardMetadata::from(expected) {
            return Err(EmbeddingError::Invalid("embedding shard identity"));
        }
        Ok(shard)
    }

    #[must_use]
    pub fn metadata(&self) -> EmbeddingShardMetadata {
        self.metadata
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn vectors(&self) -> Result<EmbeddingShardVectorReader> {
        let mut reader = BufReader::new(File::open(&self.path)?);
        reader.seek(SeekFrom::Start(u64::from(EMBEDDING_SHARD_HEADER_BYTES)))?;
        Ok(EmbeddingShardVectorReader {
            reader,
            dimensions: usize::try_from(self.metadata.dimensions)
                .map_err(|_| EmbeddingError::Invalid("embedding shard dimensions"))?,
            remaining: self.metadata.row_count,
        })
    }

    /// Validate the vectors against an embedding profile's normalization.
    /// Shape and finiteness have already been checked by `open`.
    pub fn validate_normalization(&self, normalization: &str) -> Result<()> {
        if normalization == "none" {
            return Ok(());
        }
        if normalization != "l2" {
            return Err(EmbeddingError::Invalid("embedding shard normalization"));
        }
        let dimensions = usize::try_from(self.metadata.dimensions)
            .map_err(|_| EmbeddingError::Invalid("embedding shard dimensions"))?;
        for vector in self.vectors()? {
            validate_vector(&vector?, dimensions, normalization)?;
        }
        Ok(())
    }
}

/// Verify all bytes of a task part, including its fixed identity, finite
/// values, normalization, and (when supplied) receipt-bound file digest.
pub fn verify_embedding_task_part(
    path: &Path,
    expected: EmbeddingShardExpectation,
    normalization: &str,
    expected_sha256: Option<[u8; 32]>,
) -> Result<VerifiedEmbeddingTaskPart> {
    let shard = EmbeddingShard::open_expected(path, expected)?;
    shard.validate_normalization(normalization)?;
    let sha256 = file_sha256(path)?;
    if expected_sha256.is_some_and(|expected| expected != sha256) {
        return Err(EmbeddingError::Invalid("embedding shard digest"));
    }
    Ok(VerifiedEmbeddingTaskPart {
        metadata: shard.metadata(),
        bytes: fs::metadata(path)?.len(),
        sha256,
    })
}

/// Verify a task part or atomically move an invalid part to a deterministic
/// sibling quarantine. The deterministic name makes a crash immediately after
/// quarantine discoverable on the next run.
pub fn prepare_embedding_task_part(
    path: &Path,
    expected: EmbeddingShardExpectation,
    normalization: &str,
    expected_sha256: Option<[u8; 32]>,
) -> Result<EmbeddingTaskPartPreparation> {
    if !matches!(normalization, "l2" | "none") {
        return Err(EmbeddingError::Invalid("embedding shard normalization"));
    }
    let quarantine = embedding_task_part_quarantine_path(path)?;
    let original_metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => Some(metadata),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    let quarantine_metadata = match fs::symlink_metadata(&quarantine) {
        Ok(metadata) => Some(metadata),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    if original_metadata
        .as_ref()
        .is_some_and(|metadata| !metadata.is_file() || metadata.file_type().is_symlink())
        || quarantine_metadata
            .as_ref()
            .is_some_and(|metadata| !metadata.is_file() || metadata.file_type().is_symlink())
    {
        return Err(EmbeddingError::Invalid("embedding task part path"));
    }
    let Some(_) = original_metadata else {
        return Ok(if quarantine_metadata.is_some() {
            EmbeddingTaskPartPreparation::Quarantined {
                quarantine_path: quarantine,
            }
        } else {
            EmbeddingTaskPartPreparation::Missing
        });
    };
    match verify_embedding_task_part(path, expected, normalization, expected_sha256) {
        Ok(part) => Ok(EmbeddingTaskPartPreparation::Verified {
            part,
            recovery_pending: quarantine_metadata.is_some(),
        }),
        Err(error) if quarantine_metadata.is_some() => {
            let _ = error;
            Err(EmbeddingError::Invalid(
                "embedding task part and quarantine both exist",
            ))
        }
        Err(_) => {
            fs::rename(path, &quarantine)?;
            sync_parent(path)?;
            Ok(EmbeddingTaskPartPreparation::Quarantined {
                quarantine_path: quarantine,
            })
        }
    }
}

/// Verify a newly published replacement before deleting its quarantined
/// predecessor. Higher layers should call this only after publishing the new
/// receipt, so a crash never loses the evidence needed to diagnose replacement.
pub fn complete_embedding_task_part_recovery(
    path: &Path,
    expected: EmbeddingShardExpectation,
    normalization: &str,
    expected_sha256: Option<[u8; 32]>,
) -> Result<VerifiedEmbeddingTaskPart> {
    let verified = verify_embedding_task_part(path, expected, normalization, expected_sha256)?;
    let quarantine = embedding_task_part_quarantine_path(path)?;
    match fs::symlink_metadata(&quarantine) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {
            fs::remove_file(quarantine)?;
            sync_parent(path)?;
        }
        Ok(_) => {
            return Err(EmbeddingError::Invalid(
                "embedding task part quarantine path",
            ));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(verified)
}

/// Restore a quarantined part only when no replacement exists. The restored
/// bytes are intentionally not treated as valid; callers must verify them or
/// quarantine them again before reuse.
pub fn restore_quarantined_embedding_task_part(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(_) => {
            return Err(EmbeddingError::Invalid(
                "embedding task part already exists",
            ));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let quarantine = embedding_task_part_quarantine_path(path)?;
    let metadata = fs::symlink_metadata(&quarantine)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(EmbeddingError::Invalid(
            "embedding task part quarantine path",
        ));
    }
    fs::rename(quarantine, path)?;
    sync_parent(path)
}

fn embedding_task_part_quarantine_path(path: &Path) -> Result<PathBuf> {
    let parent = path
        .parent()
        .ok_or(EmbeddingError::Invalid("embedding result parent"))?;
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or(EmbeddingError::Invalid("embedding result file name"))?;
    Ok(parent.join(format!(".{file_name}.quarantine")))
}

fn file_sha256(path: &Path) -> Result<[u8; 32]> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().into())
}

fn sync_parent(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .ok_or(EmbeddingError::Invalid("embedding result parent"))?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

pub struct EmbeddingShardVectorReader {
    reader: BufReader<File>,
    dimensions: usize,
    remaining: u64,
}

impl Iterator for EmbeddingShardVectorReader {
    type Item = Result<Vec<f32>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let byte_count = match self.dimensions.checked_mul(4) {
            Some(value) => value,
            None => return Some(Err(EmbeddingError::Invalid("embedding shard vector bytes"))),
        };
        let mut bytes = vec![0_u8; byte_count];
        if let Err(error) = self.reader.read_exact(&mut bytes) {
            self.remaining = 0;
            return Some(Err(error.into()));
        }
        let vector = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();
        self.remaining -= 1;
        if vector.iter().any(|value| !value.is_finite()) {
            Some(Err(EmbeddingError::Invalid(
                "embedding shard non-finite vector",
            )))
        } else {
            Some(Ok(vector))
        }
    }
}

pub fn decode_sha256_hex(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(EmbeddingError::Invalid("SHA-256 hex digest"));
    }
    let mut digest = [0_u8; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| EmbeddingError::Invalid("SHA-256 hex digest"))?;
    }
    Ok(digest)
}

fn encode_header(metadata: EmbeddingShardMetadata) -> [u8; 64] {
    let mut header = [0_u8; 64];
    header[0..8].copy_from_slice(&EMBEDDING_SHARD_MAGIC);
    header[8..12].copy_from_slice(&EMBEDDING_SHARD_HEADER_BYTES.to_le_bytes());
    header[12..14].copy_from_slice(&EMBEDDING_SHARD_VERSION.to_le_bytes());
    header[14] = EMBEDDING_SHARD_DTYPE_F32_LE;
    header[15] = EMBEDDING_SHARD_FLAGS;
    header[16..24].copy_from_slice(&metadata.row_count.to_le_bytes());
    header[24..28].copy_from_slice(&metadata.dimensions.to_le_bytes());
    header[28..32].copy_from_slice(&EMBEDDING_SHARD_RESERVED.to_le_bytes());
    header[32..64].copy_from_slice(&metadata.order_sha256);
    header
}

fn read_header(file: &mut File) -> Result<EmbeddingShardMetadata> {
    let mut header = [0_u8; 64];
    file.read_exact(&mut header)?;
    if header[0..8] != EMBEDDING_SHARD_MAGIC
        || u32::from_le_bytes(header[8..12].try_into().expect("four-byte field"))
            != EMBEDDING_SHARD_HEADER_BYTES
        || u16::from_le_bytes(header[12..14].try_into().expect("two-byte field"))
            != EMBEDDING_SHARD_VERSION
        || header[14] != EMBEDDING_SHARD_DTYPE_F32_LE
        || header[15] != EMBEDDING_SHARD_FLAGS
        || u32::from_le_bytes(header[28..32].try_into().expect("four-byte field"))
            != EMBEDDING_SHARD_RESERVED
    {
        return Err(EmbeddingError::Invalid("embedding shard header"));
    }
    let row_count = u64::from_le_bytes(header[16..24].try_into().expect("eight-byte field"));
    let dimensions = u32::from_le_bytes(header[24..28].try_into().expect("four-byte field"));
    if dimensions == 0 {
        return Err(EmbeddingError::Invalid("embedding shard dimensions"));
    }
    Ok(EmbeddingShardMetadata {
        row_count,
        dimensions,
        order_sha256: header[32..64].try_into().expect("32-byte digest"),
    })
}

fn validate_file_length(file: &File, metadata: EmbeddingShardMetadata) -> Result<()> {
    let payload = metadata
        .row_count
        .checked_mul(u64::from(metadata.dimensions))
        .and_then(|values| values.checked_mul(4))
        .ok_or(EmbeddingError::Invalid("embedding shard byte length"))?;
    let expected = u64::from(EMBEDDING_SHARD_HEADER_BYTES)
        .checked_add(payload)
        .ok_or(EmbeddingError::Invalid("embedding shard byte length"))?;
    if file.metadata()?.len() != expected {
        return Err(EmbeddingError::Invalid("embedding shard byte length"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::{Seek, Write};

    use tempfile::tempdir;

    use super::*;

    fn metadata() -> EmbeddingShardMetadata {
        EmbeddingShardMetadata {
            row_count: 2,
            dimensions: 2,
            order_sha256: [7; 32],
        }
    }

    #[test]
    fn writes_exact_header_and_reads_vectors() {
        let root = tempdir().unwrap();
        let path = root.path().join("part.f32");
        File::create(&path).unwrap();
        let mut writer = EmbeddingShardWriter::create(&path, metadata()).unwrap();
        writer.write_vector(&[1.0, 2.0]).unwrap();
        writer.write_vector(&[3.0, 4.0]).unwrap();
        writer.finish().unwrap();

        let shard = EmbeddingShard::open_expected(&path, metadata().into()).unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), 64 + 16);
        assert_eq!(
            shard
                .vectors()
                .unwrap()
                .collect::<Result<Vec<_>>>()
                .unwrap(),
            vec![vec![1.0, 2.0], vec![3.0, 4.0]]
        );
        let bytes = fs::read(path).unwrap();
        assert_eq!(&bytes[..8], b"LFREMB01");
    }

    #[test]
    fn rejects_corruption_wrong_identity_and_non_finite_payload() {
        let root = tempdir().unwrap();
        let path = root.path().join("part.f32");
        File::create(&path).unwrap();
        let mut writer = EmbeddingShardWriter::create(&path, metadata()).unwrap();
        writer.write_vector(&[1.0, 2.0]).unwrap();
        writer.write_vector(&[3.0, 4.0]).unwrap();
        writer.finish().unwrap();

        let wrong = EmbeddingShardExpectation {
            dimensions: 3,
            ..metadata().into()
        };
        assert!(EmbeddingShard::open_expected(&path, wrong).is_err());

        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(64)).unwrap();
        file.write_all(&f32::NAN.to_le_bytes()).unwrap();
        file.sync_all().unwrap();
        assert!(EmbeddingShard::open(&path).is_err());

        let corrupt = root.path().join("corrupt.f32");
        fs::write(&corrupt, b"too short").unwrap();
        assert!(EmbeddingShard::open(&corrupt).is_err());
    }

    #[test]
    fn publication_never_overwrites_existing_destination() {
        let root = tempdir().unwrap();
        let destination = root.path().join("part.f32");
        fs::write(&destination, b"original").unwrap();
        let publication = AtomicFilePublication::new(&destination).unwrap();
        fs::write(publication.staging_path(), b"replacement").unwrap();
        assert_eq!(
            publication.commit().unwrap(),
            AtomicPublishOutcome::AlreadyExists
        );
        assert_eq!(fs::read(destination).unwrap(), b"original");
    }

    fn write_part(path: &Path, vectors: &[[f32; 2]]) {
        File::create(path).unwrap();
        let part_metadata = EmbeddingShardMetadata {
            row_count: vectors.len() as u64,
            ..metadata()
        };
        let mut writer = EmbeddingShardWriter::create(path, part_metadata).unwrap();
        for vector in vectors {
            writer.write_vector(vector).unwrap();
        }
        writer.finish().unwrap();
    }

    #[test]
    fn invalid_task_part_is_quarantined_and_recovery_is_verified() {
        let root = tempdir().unwrap();
        let path = root.path().join("part.f32");
        let expected: EmbeddingShardExpectation = metadata().into();
        write_part(&path, &[[2.0, 0.0], [0.0, 2.0]]);

        let state = prepare_embedding_task_part(&path, expected, "l2", None).unwrap();
        let quarantine = match state {
            EmbeddingTaskPartPreparation::Quarantined { quarantine_path } => quarantine_path,
            other => panic!("expected quarantine, got {other:?}"),
        };
        assert!(!path.exists());
        assert!(quarantine.is_file());

        // A process restart rediscovers the deterministic quarantine.
        assert!(matches!(
            prepare_embedding_task_part(&path, expected, "l2", None).unwrap(),
            EmbeddingTaskPartPreparation::Quarantined { .. }
        ));

        write_part(&path, &[[1.0, 0.0], [0.0, 1.0]]);
        assert!(matches!(
            prepare_embedding_task_part(&path, expected, "l2", None).unwrap(),
            EmbeddingTaskPartPreparation::Verified {
                recovery_pending: true,
                ..
            }
        ));
        let verified = complete_embedding_task_part_recovery(&path, expected, "l2", None).unwrap();
        assert_eq!(verified.bytes, 80);
        assert!(!quarantine.exists());
        assert!(matches!(
            prepare_embedding_task_part(&path, expected, "l2", Some(verified.sha256)).unwrap(),
            EmbeddingTaskPartPreparation::Verified {
                recovery_pending: false,
                ..
            }
        ));
    }

    #[test]
    fn corrupt_task_part_can_be_restored_for_forensics_but_not_reused() {
        let root = tempdir().unwrap();
        let path = root.path().join("part.f32");
        fs::write(&path, b"corrupt").unwrap();
        let expected: EmbeddingShardExpectation = metadata().into();
        assert!(matches!(
            prepare_embedding_task_part(&path, expected, "none", None).unwrap(),
            EmbeddingTaskPartPreparation::Quarantined { .. }
        ));
        restore_quarantined_embedding_task_part(&path).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"corrupt");
        assert!(verify_embedding_task_part(&path, expected, "none", None).is_err());
        assert!(matches!(
            prepare_embedding_task_part(&path, expected, "none", None).unwrap(),
            EmbeddingTaskPartPreparation::Quarantined { .. }
        ));
    }

    #[cfg(unix)]
    #[test]
    fn task_part_recovery_rejects_symlink_paths() {
        use std::os::unix::fs::symlink;

        let root = tempdir().unwrap();
        let target = root.path().join("target");
        fs::write(&target, b"target").unwrap();
        let path = root.path().join("part.f32");
        symlink(&target, &path).unwrap();
        assert!(prepare_embedding_task_part(&path, metadata().into(), "none", None).is_err());
        assert_eq!(fs::read(target).unwrap(), b"target");
    }

    #[test]
    fn invalid_normalization_contract_does_not_quarantine_a_part() {
        let root = tempdir().unwrap();
        let path = root.path().join("part.f32");
        write_part(&path, &[[1.0, 0.0], [0.0, 1.0]]);
        assert!(prepare_embedding_task_part(&path, metadata().into(), "mystery", None).is_err());
        assert!(path.is_file());
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);
    }
}
