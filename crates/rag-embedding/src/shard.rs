use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

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
        let dimensions = usize::try_from(self.metadata.dimensions)
            .map_err(|_| EmbeddingError::Invalid("embedding shard dimensions"))?;
        for vector in self.vectors()? {
            validate_vector(&vector?, dimensions, normalization)?;
        }
        Ok(())
    }
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
}
