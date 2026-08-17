//! Exact-key artifact transfer for RunPod network volumes.
//!
//! RunPod exposes each network volume as an S3 bucket. This adapter uses only
//! `HeadObject`, `PutObject`, and `GetObject`; it never lists a bucket to decide
//! which artifacts exist. Input and task-output operations must name an object
//! already sealed in a [`CloudObjectRef`]-based transfer manifest. The sole
//! exception is a deterministic per-worker completion key: its bytes are
//! bounded and accepted only after its self digest and bundle binding pass.

use std::{
    collections::BTreeMap,
    env, fmt,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use aws_sdk_s3::{
    Client,
    config::{
        BehaviorVersion, Credentials, Region, RequestChecksumCalculation,
        ResponseChecksumValidation, retry::RetryConfig, timeout::TimeoutConfig,
    },
    primitives::{ByteStream, Length},
    types::{CompletedMultipartUpload, CompletedPart},
};
#[cfg(test)]
use rag_pipeline::Digest;
use rag_pipeline::{
    CloudObjectRef, RunpodEmbeddingBundle, RunpodTeiConformanceCandidate,
    RunpodTeiConformanceResult, RunpodWorkerAttemptMarker, SafeRelativePath,
};
use sha2::{Digest as ShaDigest, Sha256};
use tempfile::NamedTempFile;
use thiserror::Error;

const MAX_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;
/// RunPod documents single `PutObject` uploads as smaller than 500 MB.
pub const RUNPOD_SINGLE_PUT_MAX_BYTES: u64 = 500_000_000 - 1;
pub const RUNPOD_MULTIPART_PART_BYTES: u64 = 64 * 1024 * 1024;
const MAX_MULTIPART_PARTS: u64 = 10_000;
const DEFAULT_OPERATION_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const DEFAULT_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const DEFAULT_MAX_ATTEMPTS: u32 = 3;

#[derive(Debug, Error)]
pub enum RunpodS3Error {
    #[error("invalid RunPod S3 configuration: {0}")]
    Invalid(&'static str),
    #[error("RunPod S3 credential is absent from environment variable {environment}")]
    MissingCredential { environment: String },
    #[error("object key is not declared by the sealed transfer manifest")]
    ObjectNotDeclared,
    #[error("object identity differs from the sealed transfer manifest")]
    ObjectIdentityMismatch,
    #[error("local artifact could not be opened or read safely")]
    LocalRead,
    #[error("local artifact byte count or SHA-256 digest differs from its manifest")]
    LocalIdentityMismatch,
    #[error("local destination already exists")]
    DestinationExists,
    #[error("remote object is absent")]
    RemoteObjectMissing,
    #[error("remote object byte count or digest metadata differs from its manifest")]
    RemoteIdentityMismatch,
    #[error("RunPod S3 {operation} failed with HTTP status {status:?}")]
    Remote {
        operation: &'static str,
        status: Option<u16>,
    },
    #[error("downloaded artifact byte count or SHA-256 digest differs from its manifest")]
    DownloadIdentityMismatch,
    #[error("downloaded artifact could not be written atomically")]
    LocalWrite,
    #[error("completion marker is absent, too large, or fails its sealed bundle binding")]
    InvalidCompletionMarker,
    #[error(
        "conformance result is absent, too large, non-canonical, or fails its candidate binding"
    )]
    InvalidConformanceResult,
    #[error("RunPod S3 multipart upload failed during {stage}; abort {abort}")]
    MultipartFailed {
        stage: &'static str,
        abort: MultipartAbortOutcome,
    },
}

pub type Result<T> = std::result::Result<T, RunpodS3Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultipartAbortOutcome {
    Succeeded,
    Failed,
}

impl fmt::Display for MultipartAbortOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RunpodS3Limits {
    pub operation_timeout: Duration,
    pub attempt_timeout: Duration,
    pub maximum_attempts: u32,
}

impl Default for RunpodS3Limits {
    fn default() -> Self {
        Self {
            operation_timeout: DEFAULT_OPERATION_TIMEOUT,
            attempt_timeout: DEFAULT_ATTEMPT_TIMEOUT,
            maximum_attempts: DEFAULT_MAX_ATTEMPTS,
        }
    }
}

/// A sealed allowlist beneath one run-specific storage prefix.
#[derive(Debug, Clone)]
pub struct RunpodS3Manifest {
    run_prefix: SafeRelativePath,
    objects: BTreeMap<SafeRelativePath, CloudObjectRef>,
}

impl RunpodS3Manifest {
    pub fn new(
        run_prefix: SafeRelativePath,
        objects: impl IntoIterator<Item = CloudObjectRef>,
    ) -> Result<Self> {
        let mut declared = BTreeMap::new();
        for object in objects {
            validate_object(&object)?;
            if declared.insert(object.key.clone(), object).is_some() {
                return Err(RunpodS3Error::Invalid("duplicate object key"));
            }
        }
        if declared.is_empty() {
            return Err(RunpodS3Error::Invalid("empty transfer manifest"));
        }
        Ok(Self {
            run_prefix,
            objects: declared,
        })
    }

    /// Build the input allowlist from the exact object references in a cloud
    /// bundle. The caller must first validate the bundle against its prepared
    /// corpus and embedding plan.
    #[allow(dead_code)] // Used when this test-only adapter is wired into the CLI.
    pub fn from_bundle_inputs(
        run_prefix: SafeRelativePath,
        bundle: &RunpodEmbeddingBundle,
    ) -> Result<Self> {
        let artifacts = &bundle.artifacts;
        let mut objects = vec![
            artifacts.prepared_manifest.object.clone(),
            artifacts.embedding_plan.object.clone(),
            artifacts.document_token_counts.clone(),
            artifacts.embedding_profile.object.clone(),
            artifacts.executable_tokenizer.object.clone(),
            artifacts.conformance_fixture.clone(),
            artifacts.worker_binary.object.clone(),
            artifacts.model_manifest.object.clone(),
        ];
        objects.extend(artifacts.model_objects.iter().cloned());
        objects.extend(
            artifacts
                .prepared_documents
                .iter()
                .map(|artifact| artifact.object.clone()),
        );
        Self::new(run_prefix, objects)
    }

    fn declared(&self, requested: &CloudObjectRef) -> Result<&CloudObjectRef> {
        match self.objects.get(&requested.key) {
            None => Err(RunpodS3Error::ObjectNotDeclared),
            Some(declared) if declared != requested => Err(RunpodS3Error::ObjectIdentityMismatch),
            Some(declared) => Ok(declared),
        }
    }

    fn storage_key(&self, object: &CloudObjectRef) -> String {
        format!("{}/{}", self.run_prefix, object.key)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadObjectState {
    Missing,
    Present,
}

pub struct RunpodS3Client {
    client: Client,
    bucket: String,
    access_key_environment: String,
    secret_key_environment: String,
}

impl fmt::Debug for RunpodS3Client {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RunpodS3Client")
            .field("bucket", &self.bucket)
            .field("access_key_environment", &self.access_key_environment)
            .field("secret_key_environment", &self.secret_key_environment)
            .field("credentials", &"<redacted>")
            .finish()
    }
}

impl RunpodS3Client {
    /// Read the RunPod S3 access key and secret from two explicitly named
    /// environment variables. These are separate from the RunPod REST API key.
    pub fn from_environment(
        network_volume_id: &str,
        datacenter_id: &str,
        access_key_environment: &str,
        secret_key_environment: &str,
        limits: RunpodS3Limits,
    ) -> Result<Self> {
        validate_identifier(network_volume_id)?;
        validate_identifier(datacenter_id)?;
        validate_environment_name(access_key_environment)?;
        validate_environment_name(secret_key_environment)?;
        if access_key_environment == secret_key_environment {
            return Err(RunpodS3Error::Invalid("credential environment names"));
        }
        let access_key = read_credential(access_key_environment)?;
        let secret_key = read_credential(secret_key_environment)?;
        let endpoint = format!(
            "https://s3api-{}.runpod.io/",
            datacenter_id.to_ascii_lowercase()
        );
        Self::from_credentials_at(
            network_volume_id,
            datacenter_id,
            access_key_environment,
            secret_key_environment,
            &access_key,
            &secret_key,
            limits,
            &endpoint,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_credentials_at(
        network_volume_id: &str,
        datacenter_id: &str,
        access_key_environment: &str,
        secret_key_environment: &str,
        access_key: &str,
        secret_key: &str,
        limits: RunpodS3Limits,
        endpoint: &str,
        allow_http_for_test: bool,
    ) -> Result<Self> {
        validate_identifier(network_volume_id)?;
        validate_identifier(datacenter_id)?;
        validate_environment_name(access_key_environment)?;
        validate_environment_name(secret_key_environment)?;
        validate_limits(limits)?;
        validate_credential(access_key)?;
        validate_credential(secret_key)?;
        if access_key_environment == secret_key_environment {
            return Err(RunpodS3Error::Invalid("credential environment names"));
        }
        let parsed =
            reqwest::Url::parse(endpoint).map_err(|_| RunpodS3Error::Invalid("S3 endpoint"))?;
        if parsed.query().is_some()
            || parsed.fragment().is_some()
            || !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.path() != "/"
            || parsed.host_str().is_none()
            || parsed.scheme() != "https" && !(allow_http_for_test && parsed.scheme() == "http")
        {
            return Err(RunpodS3Error::Invalid("S3 endpoint"));
        }
        if !allow_http_for_test {
            let expected_host = format!("s3api-{}.runpod.io", datacenter_id.to_ascii_lowercase());
            if parsed.host_str() != Some(expected_host.as_str()) {
                return Err(RunpodS3Error::Invalid("RunPod S3 endpoint identity"));
            }
        }

        let credentials =
            Credentials::new(access_key, secret_key, None, None, "runpod-s3-environment");
        let timeout = TimeoutConfig::builder()
            .operation_timeout(limits.operation_timeout)
            .operation_attempt_timeout(limits.attempt_timeout)
            .build();
        let config = aws_sdk_s3::Config::builder()
            .behavior_version(BehaviorVersion::latest())
            .credentials_provider(credentials)
            .region(Region::new(datacenter_id.to_owned()))
            .endpoint_url(endpoint)
            .force_path_style(true)
            .retry_config(RetryConfig::standard().with_max_attempts(limits.maximum_attempts))
            .timeout_config(timeout)
            .request_checksum_calculation(RequestChecksumCalculation::WhenRequired)
            .response_checksum_validation(ResponseChecksumValidation::WhenRequired)
            .build();
        Ok(Self {
            client: Client::from_conf(config),
            bucket: network_volume_id.to_owned(),
            access_key_environment: access_key_environment.to_owned(),
            secret_key_environment: secret_key_environment.to_owned(),
        })
    }

    pub async fn head_object(
        &self,
        manifest: &RunpodS3Manifest,
        requested: &CloudObjectRef,
    ) -> Result<HeadObjectState> {
        let object = manifest.declared(requested)?;
        self.head_declared(manifest, object).await
    }

    /// Check one deterministic worker-filesystem key without listing and
    /// without requiring S3 user metadata, which mounted writes do not have.
    pub async fn head_worker_key(
        &self,
        run_prefix: &SafeRelativePath,
        key: &SafeRelativePath,
    ) -> Result<HeadObjectState> {
        let output = self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(format!("{run_prefix}/{key}"))
            .send()
            .await;
        match output {
            Ok(output) => {
                if output
                    .content_length()
                    .and_then(|value| u64::try_from(value).ok())
                    .is_none_or(|bytes| bytes == 0 || bytes > MAX_SAFE_INTEGER)
                {
                    Err(RunpodS3Error::RemoteIdentityMismatch)
                } else {
                    Ok(HeadObjectState::Present)
                }
            }
            Err(error) => {
                let status = error
                    .raw_response()
                    .map(|response| response.status().as_u16());
                if status == Some(404) {
                    Ok(HeadObjectState::Missing)
                } else {
                    Err(RunpodS3Error::Remote {
                        operation: "HEAD worker key",
                        status,
                    })
                }
            }
        }
    }

    /// Upload the local file at `local_root/object.key`. Files smaller than
    /// RunPod's 500 MB `PutObject` limit use a conditional single request;
    /// larger files use fixed-size, replayable multipart requests.
    pub async fn put_object(
        &self,
        manifest: &RunpodS3Manifest,
        requested: &CloudObjectRef,
        local_root: &Path,
    ) -> Result<()> {
        self.put_object_with_policy(
            manifest,
            requested,
            local_root,
            RUNPOD_SINGLE_PUT_MAX_BYTES,
            RUNPOD_MULTIPART_PART_BYTES,
        )
        .await
    }

    async fn put_object_with_policy(
        &self,
        manifest: &RunpodS3Manifest,
        requested: &CloudObjectRef,
        local_root: &Path,
        single_put_max_bytes: u64,
        multipart_part_bytes: u64,
    ) -> Result<()> {
        let object = manifest.declared(requested)?;
        if self.head_declared(manifest, object).await? == HeadObjectState::Present {
            return Err(RunpodS3Error::DestinationExists);
        }
        let path = resolve_existing_file(local_root, &object.key)?;
        verify_file(&path, object)?;
        if object.bytes <= single_put_max_bytes {
            self.put_single(manifest, object, &path).await
        } else {
            self.put_multipart(manifest, object, &path, multipart_part_bytes)
                .await
        }
    }

    async fn put_single(
        &self,
        manifest: &RunpodS3Manifest,
        object: &CloudObjectRef,
        path: &Path,
    ) -> Result<()> {
        let body = ByteStream::from_path(&path)
            .await
            .map_err(|_| RunpodS3Error::LocalRead)?;
        let result = self
            .client
            .put_object()
            .bucket(&self.bucket)
            .key(manifest.storage_key(object))
            .if_none_match("*")
            .metadata("sha256", object.sha256.as_str())
            .content_length(
                i64::try_from(object.bytes)
                    .map_err(|_| RunpodS3Error::Invalid("object byte length"))?,
            )
            .body(body)
            .send()
            .await;
        if let Err(error) = result {
            let status = error
                .raw_response()
                .map(|response| response.status().as_u16());
            return if matches!(status, Some(409 | 412)) {
                Err(RunpodS3Error::DestinationExists)
            } else {
                Err(RunpodS3Error::Remote {
                    operation: "PUT",
                    status,
                })
            };
        }
        match self.head_declared(manifest, object).await? {
            HeadObjectState::Present => Ok(()),
            HeadObjectState::Missing => Err(RunpodS3Error::RemoteObjectMissing),
        }
    }

    async fn put_multipart(
        &self,
        manifest: &RunpodS3Manifest,
        object: &CloudObjectRef,
        path: &Path,
        part_bytes: u64,
    ) -> Result<()> {
        // RunPod implements multipart completion but does not document an
        // atomic If-None-Match condition for it. The preliminary HEAD and the
        // run-unique prefix prevent normal reuse; a separate writer racing the
        // same sealed run key remains detectable only by post-complete identity.
        if part_bytes == 0 {
            return Err(RunpodS3Error::Invalid("multipart part size"));
        }
        let part_count = object.bytes.div_ceil(part_bytes);
        if part_count == 0 || part_count > MAX_MULTIPART_PARTS {
            return Err(RunpodS3Error::Invalid("multipart part count"));
        }
        let key = manifest.storage_key(object);
        let created = self
            .client
            .create_multipart_upload()
            .bucket(&self.bucket)
            .key(&key)
            .metadata("sha256", object.sha256.as_str())
            .send()
            .await
            .map_err(|error| RunpodS3Error::Remote {
                operation: "CREATE multipart upload",
                status: error
                    .raw_response()
                    .map(|response| response.status().as_u16()),
            })?;
        let upload_id = created
            .upload_id()
            .filter(|value| valid_opaque_response_value(value))
            .ok_or(RunpodS3Error::Remote {
                operation: "CREATE multipart upload response",
                status: Some(200),
            })?
            .to_owned();
        if created.bucket() != Some(self.bucket.as_str()) || created.key() != Some(key.as_str()) {
            return Err(self
                .multipart_failure(&key, &upload_id, "CREATE identity")
                .await);
        }

        let mut completed_parts = Vec::with_capacity(part_count as usize);
        for part_index in 0..part_count {
            let offset = part_index * part_bytes;
            let length = (object.bytes - offset).min(part_bytes);
            let part_number = i32::try_from(part_index + 1)
                .map_err(|_| RunpodS3Error::Invalid("multipart part number"))?;
            let body = match ByteStream::read_from()
                .path(path)
                .offset(offset)
                .length(Length::Exact(length))
                .buffer_size(1024 * 1024)
                .build()
                .await
            {
                Ok(body) => body,
                Err(_) => {
                    return Err(self.multipart_failure(&key, &upload_id, "read part").await);
                }
            };
            let uploaded = self
                .client
                .upload_part()
                .bucket(&self.bucket)
                .key(&key)
                .upload_id(&upload_id)
                .part_number(part_number)
                .content_length(
                    i64::try_from(length)
                        .map_err(|_| RunpodS3Error::Invalid("multipart part length"))?,
                )
                .body(body)
                .send()
                .await;
            let uploaded = match uploaded {
                Ok(uploaded) => uploaded,
                Err(_) => {
                    return Err(self
                        .multipart_failure(&key, &upload_id, "upload part")
                        .await);
                }
            };
            let Some(etag) = uploaded
                .e_tag()
                .filter(|value| valid_opaque_response_value(value))
            else {
                return Err(self.multipart_failure(&key, &upload_id, "part ETag").await);
            };
            completed_parts.push(
                CompletedPart::builder()
                    .part_number(part_number)
                    .e_tag(etag)
                    .build(),
            );
        }

        let completed = self
            .client
            .complete_multipart_upload()
            .bucket(&self.bucket)
            .key(&key)
            .upload_id(&upload_id)
            .multipart_upload(
                CompletedMultipartUpload::builder()
                    .set_parts(Some(completed_parts))
                    .build(),
            )
            .send()
            .await;
        let completed = match completed {
            Ok(completed) => completed,
            Err(_) => {
                return Err(self
                    .multipart_failure(&key, &upload_id, "complete upload")
                    .await);
            }
        };
        if completed.bucket() != Some(self.bucket.as_str())
            || completed.key() != Some(key.as_str())
            || !completed.e_tag().is_some_and(valid_opaque_response_value)
        {
            return Err(self
                .multipart_failure(&key, &upload_id, "completion identity")
                .await);
        }
        if !matches!(
            self.head_declared(manifest, object).await,
            Ok(HeadObjectState::Present)
        ) {
            return Err(self
                .multipart_failure(&key, &upload_id, "post-complete identity")
                .await);
        }
        Ok(())
    }

    async fn multipart_failure(
        &self,
        key: &str,
        upload_id: &str,
        stage: &'static str,
    ) -> RunpodS3Error {
        let abort = if self
            .client
            .abort_multipart_upload()
            .bucket(&self.bucket)
            .key(key)
            .upload_id(upload_id)
            .send()
            .await
            .is_ok()
        {
            MultipartAbortOutcome::Succeeded
        } else {
            MultipartAbortOutcome::Failed
        };
        RunpodS3Error::MultipartFailed { stage, abort }
    }

    /// Download to `destination_root/object.key`. A temporary file in the same
    /// directory is verified and persisted without overwriting an existing file.
    #[allow(dead_code)] // Retained for metadata-bearing host-upload round-trip tests.
    pub async fn get_object(
        &self,
        manifest: &RunpodS3Manifest,
        requested: &CloudObjectRef,
        destination_root: &Path,
    ) -> Result<PathBuf> {
        self.get_declared_object(manifest, requested, destination_root, true)
            .await
    }

    /// Download an exact output produced through the mounted filesystem.
    /// Such files do not carry S3 user metadata, so their declared byte count
    /// and SHA-256 digest are verified while streaming before publication.
    pub async fn get_worker_output(
        &self,
        manifest: &RunpodS3Manifest,
        requested: &CloudObjectRef,
        destination_root: &Path,
    ) -> Result<PathBuf> {
        self.get_declared_object(manifest, requested, destination_root, false)
            .await
    }

    async fn get_declared_object(
        &self,
        manifest: &RunpodS3Manifest,
        requested: &CloudObjectRef,
        destination_root: &Path,
        require_sha256_metadata: bool,
    ) -> Result<PathBuf> {
        let object = manifest.declared(requested)?;
        let destination = prepare_destination(destination_root, &object.key)?;
        if destination.exists() {
            return Err(RunpodS3Error::DestinationExists);
        }
        let output = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(manifest.storage_key(object))
            .send()
            .await
            .map_err(|error| {
                let status = error
                    .raw_response()
                    .map(|response| response.status().as_u16());
                if status == Some(404) {
                    RunpodS3Error::RemoteObjectMissing
                } else {
                    RunpodS3Error::Remote {
                        operation: "GET",
                        status,
                    }
                }
            })?;
        if output
            .content_length()
            .and_then(|value| u64::try_from(value).ok())
            != Some(object.bytes)
            || require_sha256_metadata
                && output
                    .metadata()
                    .and_then(|metadata| metadata.get("sha256"))
                    .is_none_or(|digest| digest != object.sha256.as_str())
        {
            return Err(RunpodS3Error::RemoteIdentityMismatch);
        }

        let parent = destination.parent().ok_or(RunpodS3Error::LocalWrite)?;
        let mut temporary = NamedTempFile::new_in(parent).map_err(|_| RunpodS3Error::LocalWrite)?;
        let mut body = output.body;
        let mut hasher = Sha256::new();
        let mut bytes = 0_u64;
        while let Some(chunk) = body.try_next().await.map_err(|_| RunpodS3Error::Remote {
            operation: "GET body",
            status: None,
        })? {
            bytes = bytes
                .checked_add(chunk.len() as u64)
                .ok_or(RunpodS3Error::DownloadIdentityMismatch)?;
            if bytes > object.bytes {
                return Err(RunpodS3Error::DownloadIdentityMismatch);
            }
            hasher.update(&chunk);
            temporary
                .write_all(&chunk)
                .map_err(|_| RunpodS3Error::LocalWrite)?;
        }
        temporary.flush().map_err(|_| RunpodS3Error::LocalWrite)?;
        if bytes != object.bytes || format!("{:x}", hasher.finalize()) != object.sha256.as_str() {
            return Err(RunpodS3Error::DownloadIdentityMismatch);
        }
        temporary
            .as_file()
            .sync_all()
            .map_err(|_| RunpodS3Error::LocalWrite)?;
        temporary.persist_noclobber(&destination).map_err(|error| {
            if error.error.kind() == std::io::ErrorKind::AlreadyExists {
                RunpodS3Error::DestinationExists
            } else {
                RunpodS3Error::LocalWrite
            }
        })?;
        Ok(destination)
    }

    /// Fetch one deterministic worker completion marker without listing the
    /// bucket. Mounted-filesystem outputs have no S3 user metadata, so the
    /// marker is accepted only after its canonical self digest and sealed
    /// bundle binding are proved before local publication.
    pub async fn get_completion_marker(
        &self,
        run_prefix: SafeRelativePath,
        bundle: &RunpodEmbeddingBundle,
        worker_id: &str,
        destination_root: &Path,
    ) -> Result<(RunpodWorkerAttemptMarker, PathBuf)> {
        if !bundle
            .assignments
            .iter()
            .any(|assignment| assignment.worker_id == worker_id)
        {
            return Err(RunpodS3Error::ObjectNotDeclared);
        }
        let key = SafeRelativePath::new(format!("attempts/{worker_id}/completed.json"))
            .map_err(|_| RunpodS3Error::Invalid("completion marker key"))?;
        let destination = prepare_destination(destination_root, &key)?;
        if destination.exists() {
            return Err(RunpodS3Error::DestinationExists);
        }
        let output = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(format!("{run_prefix}/{key}"))
            .send()
            .await
            .map_err(|error| RunpodS3Error::Remote {
                operation: "GET completion marker",
                status: error
                    .raw_response()
                    .map(|response| response.status().as_u16()),
            })?;
        const MAX_MARKER_BYTES: u64 = 16 * 1024 * 1024;
        let declared_bytes = output
            .content_length()
            .and_then(|value| u64::try_from(value).ok())
            .filter(|bytes| (1..=MAX_MARKER_BYTES).contains(bytes))
            .ok_or(RunpodS3Error::InvalidCompletionMarker)?;
        let metadata_sha256 = output
            .metadata()
            .and_then(|metadata| metadata.get("sha256"))
            .cloned();
        let parent = destination.parent().ok_or(RunpodS3Error::LocalWrite)?;
        let mut temporary = NamedTempFile::new_in(parent).map_err(|_| RunpodS3Error::LocalWrite)?;
        let mut body = output.body;
        let mut bytes = Vec::with_capacity(
            usize::try_from(declared_bytes).map_err(|_| RunpodS3Error::InvalidCompletionMarker)?,
        );
        while let Some(chunk) = body.try_next().await.map_err(|_| RunpodS3Error::Remote {
            operation: "GET completion marker body",
            status: None,
        })? {
            if bytes
                .len()
                .checked_add(chunk.len())
                .is_none_or(|length| length > MAX_MARKER_BYTES as usize)
            {
                return Err(RunpodS3Error::InvalidCompletionMarker);
            }
            bytes.extend_from_slice(&chunk);
        }
        let observed_sha256 = format!("{:x}", Sha256::digest(&bytes));
        if bytes.len() as u64 != declared_bytes
            || metadata_sha256
                .as_deref()
                .is_some_and(|value| value != observed_sha256)
        {
            return Err(RunpodS3Error::InvalidCompletionMarker);
        }
        let marker: RunpodWorkerAttemptMarker =
            serde_json::from_slice(&bytes).map_err(|_| RunpodS3Error::InvalidCompletionMarker)?;
        marker
            .validate_against(bundle)
            .map_err(|_| RunpodS3Error::InvalidCompletionMarker)?;
        let canonical = marker
            .canonical_object()
            .map_err(|_| RunpodS3Error::InvalidCompletionMarker)?;
        if canonical.key != key
            || canonical.bytes != declared_bytes
            || canonical.sha256.as_str() != observed_sha256
        {
            return Err(RunpodS3Error::InvalidCompletionMarker);
        }
        temporary
            .write_all(&bytes)
            .map_err(|_| RunpodS3Error::LocalWrite)?;
        temporary.flush().map_err(|_| RunpodS3Error::LocalWrite)?;
        temporary
            .as_file()
            .sync_all()
            .map_err(|_| RunpodS3Error::LocalWrite)?;
        temporary.persist_noclobber(&destination).map_err(|error| {
            if error.error.kind() == std::io::ErrorKind::AlreadyExists {
                RunpodS3Error::DestinationExists
            } else {
                RunpodS3Error::LocalWrite
            }
        })?;
        Ok((marker, destination))
    }

    /// Fetch one known conformance result without listing. The result key is
    /// derived from the caller-selected run ID, and the canonical JSON must
    /// validate against the sealed candidate before local publication.
    pub async fn get_conformance_result(
        &self,
        run_prefix: SafeRelativePath,
        candidate: &RunpodTeiConformanceCandidate,
        run_id: &str,
        destination_root: &Path,
    ) -> Result<(RunpodTeiConformanceResult, PathBuf)> {
        let key = SafeRelativePath::new(format!("conformance/results/{run_id}.json"))
            .map_err(|_| RunpodS3Error::Invalid("conformance result key"))?;
        let destination = prepare_destination(destination_root, &key)?;
        if destination.exists() {
            return Err(RunpodS3Error::DestinationExists);
        }
        let output = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(format!("{run_prefix}/{key}"))
            .send()
            .await
            .map_err(|error| RunpodS3Error::Remote {
                operation: "GET conformance result",
                status: error
                    .raw_response()
                    .map(|response| response.status().as_u16()),
            })?;
        const MAX_RESULT_BYTES: usize = 16 * 1024 * 1024;
        let declared_bytes = output
            .content_length()
            .and_then(|value| usize::try_from(value).ok())
            .filter(|bytes| (1..=MAX_RESULT_BYTES).contains(bytes))
            .ok_or(RunpodS3Error::InvalidConformanceResult)?;
        let mut body = output.body;
        let mut bytes = Vec::with_capacity(declared_bytes);
        while let Some(chunk) = body.try_next().await.map_err(|_| RunpodS3Error::Remote {
            operation: "GET conformance result body",
            status: None,
        })? {
            if bytes
                .len()
                .checked_add(chunk.len())
                .is_none_or(|length| length > MAX_RESULT_BYTES)
            {
                return Err(RunpodS3Error::InvalidConformanceResult);
            }
            bytes.extend_from_slice(&chunk);
        }
        let result: RunpodTeiConformanceResult =
            serde_json::from_slice(&bytes).map_err(|_| RunpodS3Error::InvalidConformanceResult)?;
        result
            .validate_against(candidate)
            .map_err(|_| RunpodS3Error::InvalidConformanceResult)?;
        let canonical = rag_pipeline::canonical_json_bytes(&result)
            .map_err(|_| RunpodS3Error::InvalidConformanceResult)?;
        if result.run_id != run_id || bytes != canonical {
            return Err(RunpodS3Error::InvalidConformanceResult);
        }
        let parent = destination.parent().ok_or(RunpodS3Error::LocalWrite)?;
        let mut temporary = NamedTempFile::new_in(parent).map_err(|_| RunpodS3Error::LocalWrite)?;
        temporary
            .write_all(&bytes)
            .map_err(|_| RunpodS3Error::LocalWrite)?;
        temporary.flush().map_err(|_| RunpodS3Error::LocalWrite)?;
        temporary
            .as_file()
            .sync_all()
            .map_err(|_| RunpodS3Error::LocalWrite)?;
        temporary.persist_noclobber(&destination).map_err(|error| {
            if error.error.kind() == std::io::ErrorKind::AlreadyExists {
                RunpodS3Error::DestinationExists
            } else {
                RunpodS3Error::LocalWrite
            }
        })?;
        Ok((result, destination))
    }

    async fn head_declared(
        &self,
        manifest: &RunpodS3Manifest,
        object: &CloudObjectRef,
    ) -> Result<HeadObjectState> {
        let output = self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(manifest.storage_key(object))
            .send()
            .await;
        let output = match output {
            Ok(output) => output,
            Err(error) => {
                let status = error
                    .raw_response()
                    .map(|response| response.status().as_u16());
                return if status == Some(404) {
                    Ok(HeadObjectState::Missing)
                } else {
                    Err(RunpodS3Error::Remote {
                        operation: "HEAD",
                        status,
                    })
                };
            }
        };
        if output
            .content_length()
            .and_then(|value| u64::try_from(value).ok())
            != Some(object.bytes)
            || output
                .metadata()
                .and_then(|metadata| metadata.get("sha256"))
                .is_none_or(|digest| digest != object.sha256.as_str())
        {
            return Err(RunpodS3Error::RemoteIdentityMismatch);
        }
        Ok(HeadObjectState::Present)
    }
}

fn validate_object(object: &CloudObjectRef) -> Result<()> {
    if object.bytes == 0 || object.bytes > MAX_SAFE_INTEGER {
        return Err(RunpodS3Error::Invalid("object byte length"));
    }
    Ok(())
}

fn validate_limits(limits: RunpodS3Limits) -> Result<()> {
    if limits.operation_timeout.is_zero()
        || limits.attempt_timeout.is_zero()
        || limits.attempt_timeout > limits.operation_timeout
        || !(1..=10).contains(&limits.maximum_attempts)
    {
        return Err(RunpodS3Error::Invalid("S3 client limits"));
    }
    Ok(())
}

fn validate_identifier(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
    {
        return Err(RunpodS3Error::Invalid("RunPod identifier"));
    }
    Ok(())
}

fn validate_environment_name(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_')
    {
        return Err(RunpodS3Error::Invalid("environment variable name"));
    }
    Ok(())
}

fn validate_credential(value: &str) -> Result<()> {
    if value.is_empty() || value.len() > 4096 || value.chars().any(char::is_whitespace) {
        return Err(RunpodS3Error::Invalid("S3 credential"));
    }
    Ok(())
}

fn valid_opaque_response_value(value: &str) -> bool {
    !value.is_empty() && value.len() <= 1024 && !value.chars().any(char::is_control)
}

fn read_credential(environment: &str) -> Result<String> {
    env::var(environment)
        .ok()
        .filter(|value| validate_credential(value).is_ok())
        .ok_or_else(|| RunpodS3Error::MissingCredential {
            environment: environment.to_owned(),
        })
}

fn resolve_existing_file(root: &Path, key: &SafeRelativePath) -> Result<PathBuf> {
    let root = root.canonicalize().map_err(|_| RunpodS3Error::LocalRead)?;
    let candidate = key
        .join_to(&root)
        .canonicalize()
        .map_err(|_| RunpodS3Error::LocalRead)?;
    if !candidate.starts_with(&root) || !candidate.is_file() {
        return Err(RunpodS3Error::LocalRead);
    }
    Ok(candidate)
}

fn prepare_destination(root: &Path, key: &SafeRelativePath) -> Result<PathBuf> {
    let root = root.canonicalize().map_err(|_| RunpodS3Error::LocalWrite)?;
    if !root.is_dir() {
        return Err(RunpodS3Error::LocalWrite);
    }
    let components = key.as_str().split('/').collect::<Vec<_>>();
    let mut parent = root.clone();
    for component in &components[..components.len() - 1] {
        parent.push(component);
        if !parent.exists() {
            fs::create_dir(&parent).map_err(|_| RunpodS3Error::LocalWrite)?;
        }
        parent = parent
            .canonicalize()
            .map_err(|_| RunpodS3Error::LocalWrite)?;
        if !parent.starts_with(&root) || !parent.is_dir() {
            return Err(RunpodS3Error::LocalWrite);
        }
    }
    Ok(parent.join(components.last().expect("safe paths are nonempty")))
}

fn verify_file(path: &Path, expected: &CloudObjectRef) -> Result<()> {
    let mut file = File::open(path).map_err(|_| RunpodS3Error::LocalRead)?;
    let metadata = file.metadata().map_err(|_| RunpodS3Error::LocalRead)?;
    if !metadata.is_file() || metadata.len() != expected.bytes {
        return Err(RunpodS3Error::LocalIdentityMismatch);
    }
    let mut hasher = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|_| RunpodS3Error::LocalRead)?;
        if count == 0 {
            break;
        }
        bytes = bytes
            .checked_add(count as u64)
            .ok_or(RunpodS3Error::LocalIdentityMismatch)?;
        hasher.update(&buffer[..count]);
    }
    if bytes != expected.bytes || format!("{:x}", hasher.finalize()) != expected.sha256.as_str() {
        return Err(RunpodS3Error::LocalIdentityMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        io::{BufRead, BufReader},
        net::{TcpListener, TcpStream},
        sync::mpsc,
        thread,
    };

    use super::*;

    const ACCESS: &str = "user_fake_access";
    const SECRET: &str = "rps_fake_secret_value";
    const BODY: &[u8] = b"sealed artifact bytes";

    #[derive(Clone)]
    struct FakeResponse {
        status: u16,
        body: Vec<u8>,
        headers: Vec<(String, String)>,
        declared_length: Option<usize>,
    }

    impl FakeResponse {
        fn new(status: u16, body: impl Into<Vec<u8>>) -> Self {
            Self {
                status,
                body: body.into(),
                headers: Vec::new(),
                declared_length: None,
            }
        }

        fn object(body: impl Into<Vec<u8>>) -> Self {
            let body = body.into();
            Self::new(200, body.clone())
                .header("x-amz-meta-sha256", &sha256(&body))
                .declared_length(body.len())
        }

        fn header(mut self, name: &str, value: &str) -> Self {
            self.headers.push((name.into(), value.into()));
            self
        }

        fn declared_length(mut self, length: usize) -> Self {
            self.declared_length = Some(length);
            self
        }
    }

    fn spawn_server(
        responses: Vec<FakeResponse>,
    ) -> (String, mpsc::Receiver<Vec<u8>>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, receiver) = mpsc::channel();
        let handle = thread::spawn(move || {
            for response in responses {
                let (mut stream, _) = listener.accept().unwrap();
                let request = read_request(&mut stream);
                let _ = sender.send(request);
                let reason = match response.status {
                    200 => "OK",
                    404 => "Not Found",
                    409 => "Conflict",
                    412 => "Precondition Failed",
                    500 => "Internal Server Error",
                    502 => "Bad Gateway",
                    _ => "Status",
                };
                let length = response.declared_length.unwrap_or(response.body.len());
                write!(
                    stream,
                    "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nConnection: close\r\n",
                    response.status, reason, length
                )
                .unwrap();
                for (name, value) in response.headers {
                    write!(stream, "{name}: {value}\r\n").unwrap();
                }
                write!(stream, "\r\n").unwrap();
                stream.write_all(&response.body).unwrap();
            }
        });
        (format!("http://{address}/"), receiver, handle)
    }

    fn read_request(stream: &mut TcpStream) -> Vec<u8> {
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut header = Vec::new();
        loop {
            let mut line = Vec::new();
            reader.read_until(b'\n', &mut line).unwrap();
            assert!(!line.is_empty());
            let done = line == b"\r\n";
            header.extend(line);
            if done {
                break;
            }
        }
        let header_text = String::from_utf8_lossy(&header);
        let length = header_text
            .lines()
            .find_map(|line| {
                line.to_ascii_lowercase()
                    .strip_prefix("content-length: ")
                    .and_then(|value| value.parse::<usize>().ok())
            })
            .unwrap_or(0);
        let mut body = vec![0_u8; length];
        reader.read_exact(&mut body).unwrap();
        header.extend(body);
        header
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn object(bytes: &[u8]) -> CloudObjectRef {
        CloudObjectRef {
            key: SafeRelativePath::new("inputs/shard.parquet").unwrap(),
            bytes: bytes.len() as u64,
            sha256: Digest::new(sha256(bytes)).unwrap(),
        }
    }

    fn manifest(expected: &CloudObjectRef) -> RunpodS3Manifest {
        RunpodS3Manifest::new(
            SafeRelativePath::new("runs/test-run").unwrap(),
            [expected.clone()],
        )
        .unwrap()
    }

    fn client(endpoint: &str, maximum_attempts: u32) -> RunpodS3Client {
        RunpodS3Client::from_credentials_at(
            "volume-1",
            "US-KS-2",
            "RUNPOD_S3_ACCESS_KEY",
            "RUNPOD_S3_SECRET_KEY",
            ACCESS,
            SECRET,
            RunpodS3Limits {
                operation_timeout: Duration::from_secs(5),
                attempt_timeout: Duration::from_secs(2),
                maximum_attempts,
            },
            endpoint,
            true,
        )
        .unwrap()
    }

    fn request_text(request: &[u8]) -> String {
        String::from_utf8_lossy(request).to_ascii_lowercase()
    }

    fn create_multipart_xml(key: &str) -> Vec<u8> {
        format!(
            "<InitiateMultipartUploadResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\"><Bucket>volume-1</Bucket><Key>{key}</Key><UploadId>upload-1</UploadId></InitiateMultipartUploadResult>"
        )
        .into_bytes()
    }

    fn complete_multipart_xml(key: &str) -> Vec<u8> {
        format!(
            "<CompleteMultipartUploadResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\"><Location>http://example.invalid/{key}</Location><Bucket>volume-1</Bucket><Key>{key}</Key><ETag>&quot;complete-etag&quot;</ETag></CompleteMultipartUploadResult>"
        )
        .into_bytes()
    }

    #[tokio::test]
    async fn head_uses_exact_prefixed_key_and_header_sigv4() {
        let expected = object(BODY);
        let (endpoint, requests, server) = spawn_server(vec![
            FakeResponse::new(200, Vec::new())
                .declared_length(BODY.len())
                .header("x-amz-meta-sha256", expected.sha256.as_str()),
        ]);
        let state = client(&endpoint, 1)
            .head_object(&manifest(&expected), &expected)
            .await
            .unwrap();
        assert_eq!(state, HeadObjectState::Present);
        let request = request_text(&requests.recv().unwrap());
        assert!(request.starts_with("head /volume-1/runs/test-run/inputs/shard.parquet"));
        assert!(request.contains("authorization: aws4-hmac-sha256"));
        assert!(!request.contains(SECRET));
        assert!(!request.contains("x-amz-signature="));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn put_prechecks_local_digest_streams_and_refuses_overwrite() {
        let staging = tempfile::tempdir().unwrap();
        fs::create_dir(staging.path().join("inputs")).unwrap();
        fs::write(staging.path().join("inputs/shard.parquet"), BODY).unwrap();
        let expected = object(BODY);
        let head = FakeResponse::new(404, b"<Error><Code>NoSuchKey</Code></Error>".to_vec());
        let uploaded = FakeResponse::new(200, Vec::new());
        let verify = FakeResponse::new(200, Vec::new())
            .declared_length(BODY.len())
            .header("x-amz-meta-sha256", expected.sha256.as_str());
        let (endpoint, requests, server) = spawn_server(vec![head, uploaded, verify]);
        client(&endpoint, 1)
            .put_object(&manifest(&expected), &expected, staging.path())
            .await
            .unwrap();
        let _head = requests.recv().unwrap();
        let put = requests.recv().unwrap();
        let put_text = request_text(&put);
        assert!(put_text.starts_with("put /volume-1/runs/test-run/inputs/shard.parquet"));
        assert!(put_text.contains("if-none-match: *"));
        assert!(put_text.contains(&format!("x-amz-meta-sha256: {}", expected.sha256)));
        assert!(put.ends_with(BODY));
        let _verify = requests.recv().unwrap();
        server.join().unwrap();

        fs::write(staging.path().join("inputs/shard.parquet"), b"wrong bytes").unwrap();
        let (endpoint, _, server) = spawn_server(vec![FakeResponse::new(
            404,
            b"<Error><Code>NoSuchKey</Code></Error>".to_vec(),
        )]);
        let error = client(&endpoint, 1)
            .put_object(&manifest(&expected), &expected, staging.path())
            .await
            .unwrap_err();
        assert!(matches!(error, RunpodS3Error::LocalIdentityMismatch));
        server.join().unwrap();

        let (endpoint, requests, server) = spawn_server(vec![
            FakeResponse::new(200, Vec::new())
                .declared_length(BODY.len())
                .header("x-amz-meta-sha256", expected.sha256.as_str()),
        ]);
        let error = client(&endpoint, 1)
            .put_object(&manifest(&expected), &expected, staging.path())
            .await
            .unwrap_err();
        assert!(matches!(error, RunpodS3Error::DestinationExists));
        assert_eq!(
            requests.iter().count(),
            1,
            "an existing object must not be PUT"
        );
        server.join().unwrap();

        fs::write(staging.path().join("inputs/shard.parquet"), BODY).unwrap();
        let (endpoint, _, server) = spawn_server(vec![
            FakeResponse::new(404, Vec::new()),
            FakeResponse::new(412, Vec::new()),
        ]);
        let error = client(&endpoint, 1)
            .put_object(&manifest(&expected), &expected, staging.path())
            .await
            .unwrap_err();
        assert!(matches!(error, RunpodS3Error::DestinationExists));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn multipart_upload_streams_fixed_replayable_parts_and_completes_exactly() {
        let staging = tempfile::tempdir().unwrap();
        fs::create_dir(staging.path().join("inputs")).unwrap();
        fs::write(staging.path().join("inputs/shard.parquet"), BODY).unwrap();
        let expected = object(BODY);
        let key = "runs/test-run/inputs/shard.parquet";
        let mut responses = vec![
            FakeResponse::new(404, Vec::new()),
            FakeResponse::new(200, create_multipart_xml(key))
                .header("content-type", "application/xml"),
        ];
        for etag in ["part-1", "part-2", "part-3"] {
            responses.push(FakeResponse::new(200, Vec::new()).header("etag", etag));
        }
        responses.push(
            FakeResponse::new(200, complete_multipart_xml(key))
                .header("content-type", "application/xml"),
        );
        responses.push(
            FakeResponse::new(200, Vec::new())
                .declared_length(BODY.len())
                .header("x-amz-meta-sha256", expected.sha256.as_str()),
        );
        let (endpoint, requests, server) = spawn_server(responses);
        client(&endpoint, 1)
            .put_object_with_policy(&manifest(&expected), &expected, staging.path(), 8, 8)
            .await
            .unwrap();
        let requests = requests.iter().collect::<Vec<_>>();
        assert_eq!(requests.len(), 7);
        let create = request_text(&requests[1]);
        assert!(create.starts_with("post /volume-1/runs/test-run/inputs/shard.parquet?uploads"));
        assert!(create.contains(&format!("x-amz-meta-sha256: {}", expected.sha256)));
        for (index, request) in requests[2..5].iter().enumerate() {
            let text = request_text(request);
            assert!(text.starts_with("put /volume-1/runs/test-run/inputs/shard.parquet?"));
            assert!(text.contains(&format!("partnumber={}", index + 1)));
            assert!(text.contains("uploadid=upload-1"));
        }
        let complete = request_text(&requests[5]);
        assert!(complete.starts_with("post /volume-1/runs/test-run/inputs/shard.parquet?"));
        assert!(complete.contains("uploadid=upload-1"));
        assert!(complete.contains("<partnumber>1</partnumber>"));
        assert!(complete.contains("<partnumber>3</partnumber>"));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn multipart_part_failure_and_completion_mismatch_are_aborted() {
        let staging = tempfile::tempdir().unwrap();
        fs::create_dir(staging.path().join("inputs")).unwrap();
        fs::write(staging.path().join("inputs/shard.parquet"), BODY).unwrap();
        let expected = object(BODY);
        let key = "runs/test-run/inputs/shard.parquet";

        let (endpoint, requests, server) = spawn_server(vec![
            FakeResponse::new(404, Vec::new()),
            FakeResponse::new(200, create_multipart_xml(key))
                .header("content-type", "application/xml"),
            FakeResponse::new(500, b"<Error><Code>InternalError</Code></Error>".to_vec())
                .header("content-type", "application/xml"),
            FakeResponse::new(204, Vec::new()),
        ]);
        let error = client(&endpoint, 1)
            .put_object_with_policy(&manifest(&expected), &expected, staging.path(), 8, 8)
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            RunpodS3Error::MultipartFailed {
                stage: "upload part",
                abort: MultipartAbortOutcome::Succeeded
            }
        ));
        let requests = requests.iter().collect::<Vec<_>>();
        assert_eq!(requests.len(), 4);
        assert!(request_text(&requests[3]).starts_with("delete "));
        assert!(request_text(&requests[3]).contains("uploadid=upload-1"));
        server.join().unwrap();

        let mut responses = vec![
            FakeResponse::new(404, Vec::new()),
            FakeResponse::new(200, create_multipart_xml(key))
                .header("content-type", "application/xml"),
        ];
        for etag in ["part-1", "part-2", "part-3"] {
            responses.push(FakeResponse::new(200, Vec::new()).header("etag", etag));
        }
        responses.push(
            FakeResponse::new(200, complete_multipart_xml("runs/test-run/wrong-key"))
                .header("content-type", "application/xml"),
        );
        responses.push(FakeResponse::new(204, Vec::new()));
        let (endpoint, requests, server) = spawn_server(responses);
        let error = client(&endpoint, 1)
            .put_object_with_policy(&manifest(&expected), &expected, staging.path(), 8, 8)
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            RunpodS3Error::MultipartFailed {
                stage: "completion identity",
                abort: MultipartAbortOutcome::Succeeded
            }
        ));
        let requests = requests.iter().collect::<Vec<_>>();
        assert_eq!(requests.len(), 7);
        assert!(request_text(&requests[6]).starts_with("delete "));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn remote_sha256_metadata_is_required_for_head_and_get() {
        let expected = object(BODY);
        let (endpoint, _, server) = spawn_server(vec![
            FakeResponse::new(200, Vec::new()).declared_length(BODY.len()),
        ]);
        let error = client(&endpoint, 1)
            .head_object(&manifest(&expected), &expected)
            .await
            .unwrap_err();
        assert!(matches!(error, RunpodS3Error::RemoteIdentityMismatch));
        server.join().unwrap();

        let destination = tempfile::tempdir().unwrap();
        let (endpoint, _, server) = spawn_server(vec![FakeResponse::new(200, BODY.to_vec())]);
        let error = client(&endpoint, 1)
            .get_object(&manifest(&expected), &expected, destination.path())
            .await
            .unwrap_err();
        assert!(matches!(error, RunpodS3Error::RemoteIdentityMismatch));
        assert!(!destination.path().join(expected.key.to_string()).exists());
        server.join().unwrap();
    }

    #[tokio::test]
    async fn get_streams_to_atomic_file_and_rejects_wrong_hash() {
        let expected = object(BODY);
        let destination = tempfile::tempdir().unwrap();
        let (endpoint, requests, server) = spawn_server(vec![FakeResponse::object(BODY)]);
        let path = client(&endpoint, 1)
            .get_object(&manifest(&expected), &expected, destination.path())
            .await
            .unwrap();
        assert_eq!(fs::read(&path).unwrap(), BODY);
        let get_request = request_text(&requests.recv().unwrap());
        assert!(
            get_request.starts_with("get /volume-1/runs/test-run/inputs/shard.parquet"),
            "{get_request}"
        );
        server.join().unwrap();

        let bad_destination = tempfile::tempdir().unwrap();
        let wrong = b"sealed artifact bytez";
        assert_eq!(wrong.len(), BODY.len());
        let (endpoint, _, server) = spawn_server(vec![
            FakeResponse::new(200, wrong.to_vec())
                .header("x-amz-meta-sha256", expected.sha256.as_str()),
        ]);
        let error = client(&endpoint, 1)
            .get_object(&manifest(&expected), &expected, bad_destination.path())
            .await
            .unwrap_err();
        assert!(matches!(error, RunpodS3Error::DownloadIdentityMismatch));
        assert!(
            !bad_destination
                .path()
                .join(expected.key.to_string())
                .exists()
        );
        server.join().unwrap();
    }

    #[tokio::test]
    async fn mounted_worker_output_needs_no_s3_metadata_but_still_needs_exact_bytes() {
        let expected = object(BODY);
        let destination = tempfile::tempdir().unwrap();
        let (endpoint, requests, server) =
            spawn_server(vec![FakeResponse::new(200, BODY.to_vec())]);
        let path = client(&endpoint, 1)
            .get_worker_output(&manifest(&expected), &expected, destination.path())
            .await
            .unwrap();
        assert_eq!(fs::read(path).unwrap(), BODY);
        assert!(request_text(&requests.recv().unwrap()).starts_with("get "));
        server.join().unwrap();

        let wrong = b"sealed artifact bytez";
        let destination = tempfile::tempdir().unwrap();
        let (endpoint, _, server) = spawn_server(vec![FakeResponse::new(200, wrong.to_vec())]);
        assert!(matches!(
            client(&endpoint, 1)
                .get_worker_output(&manifest(&expected), &expected, destination.path())
                .await,
            Err(RunpodS3Error::DownloadIdentityMismatch)
        ));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn fake_server_round_trips_exact_storage_challenge_keys_without_listing() {
        let staging = tempfile::tempdir().unwrap();
        let challenge_key = SafeRelativePath::new(".storage-challenge.bin").unwrap();
        let challenge_path = challenge_key.join_to(staging.path());
        fs::create_dir_all(challenge_path.parent().unwrap()).unwrap();
        fs::write(&challenge_path, BODY).unwrap();
        let challenge = CloudObjectRef {
            key: challenge_key,
            bytes: BODY.len() as u64,
            sha256: Digest::new(sha256(BODY)).unwrap(),
        };
        let challenge_manifest = RunpodS3Manifest::new(
            SafeRelativePath::new("runs/storage-proof").unwrap(),
            [challenge.clone()],
        )
        .unwrap();
        let (endpoint, requests, server) = spawn_server(vec![
            FakeResponse::new(404, Vec::new()),
            FakeResponse::new(200, Vec::new()),
            FakeResponse::new(200, Vec::new())
                .declared_length(BODY.len())
                .header("x-amz-meta-sha256", challenge.sha256.as_str()),
        ]);
        client(&endpoint, 1)
            .put_object(&challenge_manifest, &challenge, staging.path())
            .await
            .unwrap();
        let challenge_requests = requests.iter().collect::<Vec<_>>();
        assert_eq!(challenge_requests.len(), 3);
        assert!(
            request_text(&challenge_requests[1])
                .starts_with("put /volume-1/runs/storage-proof/.storage-challenge.bin")
        );
        assert!(
            challenge_requests
                .iter()
                .all(|request| !request_text(request).contains("list-type"))
        );
        server.join().unwrap();

        let response_bytes = b"{\"publication\":\"hard_link_no_overwrite\"}";
        let response = CloudObjectRef {
            key: SafeRelativePath::new(".storage-challenge-response.json").unwrap(),
            bytes: response_bytes.len() as u64,
            sha256: Digest::new(sha256(response_bytes)).unwrap(),
        };
        let response_manifest = RunpodS3Manifest::new(
            SafeRelativePath::new("runs/storage-proof").unwrap(),
            [response.clone()],
        )
        .unwrap();
        let destination = tempfile::tempdir().unwrap();
        let (endpoint, requests, server) =
            spawn_server(vec![FakeResponse::new(200, response_bytes.to_vec())]);
        let path = client(&endpoint, 1)
            .get_worker_output(&response_manifest, &response, destination.path())
            .await
            .unwrap();
        assert_eq!(fs::read(path).unwrap(), response_bytes);
        let request = request_text(&requests.recv().unwrap());
        assert!(
            request
                .starts_with("get /volume-1/runs/storage-proof/.storage-challenge-response.json")
        );
        assert!(!request.contains("list-type"));
        assert!(!request.contains(SECRET));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn interrupted_body_is_not_persisted_and_retryable_status_is_retried() {
        let expected = object(BODY);
        let destination = tempfile::tempdir().unwrap();
        let partial = FakeResponse::new(200, BODY[..5].to_vec())
            .declared_length(BODY.len())
            .header("x-amz-meta-sha256", expected.sha256.as_str());
        let (endpoint, _, server) = spawn_server(vec![partial]);
        let error = client(&endpoint, 1)
            .get_object(&manifest(&expected), &expected, destination.path())
            .await
            .unwrap_err();
        assert!(matches!(error, RunpodS3Error::Remote { .. }));
        assert!(!destination.path().join(expected.key.to_string()).exists());
        server.join().unwrap();

        let retry_body = b"<Error><Code>InternalError</Code></Error>".to_vec();
        let (endpoint, requests, server) = spawn_server(vec![
            FakeResponse::new(502, retry_body),
            FakeResponse::new(200, Vec::new())
                .declared_length(BODY.len())
                .header("x-amz-meta-sha256", expected.sha256.as_str()),
        ]);
        assert_eq!(
            client(&endpoint, 2)
                .head_object(&manifest(&expected), &expected)
                .await
                .unwrap(),
            HeadObjectState::Present
        );
        assert_eq!(requests.iter().count(), 2);
        server.join().unwrap();
    }

    #[tokio::test]
    async fn missing_changed_and_extra_keys_are_never_discovered_by_listing() {
        let expected = object(BODY);
        let manifest = manifest(&expected);
        let missing = CloudObjectRef {
            key: SafeRelativePath::new("inputs/missing.parquet").unwrap(),
            ..expected.clone()
        };
        let changed = CloudObjectRef {
            bytes: expected.bytes + 1,
            ..expected.clone()
        };
        let disconnected_client = client("http://127.0.0.1:1/", 1);
        assert!(matches!(
            disconnected_client.head_object(&manifest, &missing).await,
            Err(RunpodS3Error::ObjectNotDeclared)
        ));
        assert!(matches!(
            disconnected_client.head_object(&manifest, &changed).await,
            Err(RunpodS3Error::ObjectIdentityMismatch)
        ));

        let (endpoint, requests, server) = spawn_server(vec![FakeResponse::new(404, Vec::new())]);
        assert_eq!(
            client(&endpoint, 1)
                .head_object(&manifest, &expected)
                .await
                .unwrap(),
            HeadObjectState::Missing
        );
        let request = request_text(&requests.recv().unwrap());
        assert!(request.starts_with("head "));
        assert!(!request.contains("list-type"));
        server.join().unwrap();
    }

    #[test]
    fn path_escape_https_and_secret_redaction_fail_closed() {
        assert!(SafeRelativePath::new("../escape").is_err());
        let expected = object(BODY);
        assert!(
            RunpodS3Manifest::new(
                SafeRelativePath::new("runs/test").unwrap(),
                [expected.clone(), expected]
            )
            .is_err()
        );
        assert!(
            RunpodS3Client::from_credentials_at(
                "volume-1",
                "US-KS-2",
                "RUNPOD_S3_ACCESS_KEY",
                "RUNPOD_S3_SECRET_KEY",
                ACCESS,
                SECRET,
                RunpodS3Limits::default(),
                "http://s3api-us-ks-2.runpod.io/",
                false,
            )
            .is_err()
        );
        let client = client("http://127.0.0.1:1/", 1);
        let rendered = format!("{client:?}");
        assert!(!rendered.contains(ACCESS));
        assert!(!rendered.contains(SECRET));
        assert!(rendered.contains("<redacted>"));
        for error in [
            RunpodS3Error::Remote {
                operation: "GET",
                status: Some(403),
            },
            RunpodS3Error::LocalIdentityMismatch,
        ] {
            let rendered = format!("{error:?} {error}");
            assert!(!rendered.contains(ACCESS));
            assert!(!rendered.contains(SECRET));
        }

        let missing = RunpodS3Client::from_environment(
            "volume-1",
            "US-KS-2",
            "LIVEFIRE_RUNPOD_S3_TEST_ACCESS_KEY_NOT_SET",
            "LIVEFIRE_RUNPOD_S3_TEST_SECRET_KEY_NOT_SET",
            RunpodS3Limits::default(),
        )
        .unwrap_err();
        assert!(matches!(missing, RunpodS3Error::MissingCredential { .. }));
    }

    #[cfg(unix)]
    #[test]
    fn local_symlinks_cannot_escape_the_staging_root() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("shard.parquet"), BODY).unwrap();
        symlink(outside.path(), root.path().join("inputs")).unwrap();
        let key = SafeRelativePath::new("inputs/shard.parquet").unwrap();
        assert!(matches!(
            resolve_existing_file(root.path(), &key),
            Err(RunpodS3Error::LocalRead)
        ));
        assert!(matches!(
            prepare_destination(root.path(), &key),
            Err(RunpodS3Error::LocalWrite)
        ));
    }
}
