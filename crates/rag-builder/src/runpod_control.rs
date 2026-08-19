//! Minimal, fail-closed client for the RunPod Secure Cloud REST control plane.
//!
//! This module creates and inspects compute and storage only. Artifact transfer,
//! worker execution, and S3-compatible storage are deliberately out of scope.

use std::{collections::BTreeMap, env, fmt, time::Duration};

use reqwest::{
    Client, RequestBuilder, StatusCode,
    header::{AUTHORIZATION, CONTENT_LENGTH, HeaderValue},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use thiserror::Error;

pub const RUNPOD_REST_V1: &str = "https://rest.runpod.io/v1";
pub const RUNPOD_GRAPHQL: &str = "https://api.runpod.io/graphql";
pub const WORKSPACE_MOUNT: &str = "/workspace";
pub const WORKER_IMAGE_PATH: &str = "/usr/local/bin/rag-runpod-worker";
pub const REDACTED_BEARER: &str = "Bearer <redacted>";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_MAX_RESPONSE_BYTES: usize = 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CleanupOutcome {
    Succeeded,
    Failed,
}

impl fmt::Display for CleanupOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Succeeded => "cleanup succeeded",
            Self::Failed => "cleanup failed",
        })
    }
}

#[derive(Debug, Error)]
pub enum RunpodControlError {
    #[error("invalid RunPod control-plane configuration: {0}")]
    Invalid(&'static str),
    #[error("RunPod API key is absent from environment variable {environment}")]
    MissingApiKey { environment: String },
    #[error("RunPod HTTP request failed during {operation}")]
    Transport { operation: &'static str },
    #[error("RunPod {operation} returned unexpected HTTP status {status}")]
    UnexpectedStatus {
        operation: &'static str,
        status: u16,
    },
    #[error("RunPod {operation} response exceeded {limit} bytes")]
    ResponseTooLarge {
        operation: &'static str,
        limit: usize,
    },
    #[error("RunPod {operation} response was not the declared JSON shape")]
    MalformedResponse { operation: &'static str },
    #[error("RunPod {operation} response was not the declared JSON shape: {reason}")]
    MalformedJsonResponse {
        operation: &'static str,
        reason: String,
    },
    #[error("RunPod scheduler rejected the Pod request: {reason}")]
    SchedulerRejected { reason: String },
    #[error("created RunPod Pod was rejected: {reason}; {cleanup}")]
    CreatedPodRejected {
        reason: &'static str,
        cleanup: CleanupOutcome,
    },
}

pub type Result<T> = std::result::Result<T, RunpodControlError>;

/// Client limits are local safety policy and are never sent to RunPod.
#[derive(Debug, Clone, Copy)]
pub struct RunpodClientLimits {
    pub timeout: Duration,
    pub maximum_response_bytes: usize,
}

impl Default for RunpodClientLimits {
    fn default() -> Self {
        Self {
            timeout: DEFAULT_TIMEOUT,
            maximum_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
        }
    }
}

pub struct RunpodClient {
    http: Client,
    base_url: String,
    graphql_url: String,
    authorization: HeaderValue,
    maximum_response_bytes: usize,
    api_key_environment: String,
}

impl fmt::Debug for RunpodClient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RunpodClient")
            .field("base_url", &self.base_url)
            .field("graphql_url", &self.graphql_url)
            .field("authorization", &REDACTED_BEARER)
            .field("maximum_response_bytes", &self.maximum_response_bytes)
            .field("api_key_environment", &self.api_key_environment)
            .finish()
    }
}

impl RunpodClient {
    /// Read the bearer token from one explicitly named environment variable.
    pub fn from_environment(environment: &str, limits: RunpodClientLimits) -> Result<Self> {
        Self::from_environment_at(environment, limits, RUNPOD_REST_V1)
    }

    fn from_environment_at(
        environment: &str,
        limits: RunpodClientLimits,
        base_url: &str,
    ) -> Result<Self> {
        validate_environment_name(environment)?;
        let api_key = env::var(environment)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| RunpodControlError::MissingApiKey {
                environment: environment.to_owned(),
            })?;
        Self::from_api_key_at(environment, &api_key, limits, base_url)
    }

    fn from_api_key_at(
        environment: &str,
        api_key: &str,
        limits: RunpodClientLimits,
        base_url: &str,
    ) -> Result<Self> {
        validate_environment_name(environment)?;
        validate_limits(limits)?;
        if api_key.is_empty()
            || api_key.len() > 4096
            || api_key.chars().any(char::is_whitespace)
            || !base_url.starts_with("http://") && !base_url.starts_with("https://")
        {
            return Err(RunpodControlError::Invalid("API key or base URL"));
        }
        let mut authorization = HeaderValue::from_str(&format!("Bearer {api_key}"))
            .map_err(|_| RunpodControlError::Invalid("API key header"))?;
        authorization.set_sensitive(true);
        let http = Client::builder()
            .timeout(limits.timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|_| RunpodControlError::Invalid("HTTP client"))?;
        Ok(Self {
            http,
            graphql_url: if base_url.trim_end_matches('/') == RUNPOD_REST_V1 {
                RUNPOD_GRAPHQL.to_owned()
            } else {
                format!(
                    "{}/graphql",
                    base_url.trim_end_matches('/').trim_end_matches("/v1")
                )
            },
            base_url: base_url.trim_end_matches('/').to_owned(),
            authorization,
            maximum_response_bytes: limits.maximum_response_bytes,
            api_key_environment: environment.to_owned(),
        })
    }

    #[allow(dead_code)] // Retained for adapter-level fake-server tests.
    pub fn dry_run_create_pod(&self, specification: &PodCreateSpec) -> Result<Value> {
        dry_run_create_pod_at(specification, &self.api_key_environment, &self.url("pods"))
    }
}

/// Render the complete Pod request without reading credentials or contacting
/// RunPod. The named environment variable is recorded only as a redacted
/// runtime source.
#[allow(dead_code)] // Retained for REST adapter tests; paid launches use the GraphQL TTL request.
pub fn dry_run_create_pod(
    specification: &PodCreateSpec,
    api_key_environment: &str,
) -> Result<Value> {
    validate_environment_name(api_key_environment)?;
    dry_run_create_pod_at(
        specification,
        api_key_environment,
        &format!("{RUNPOD_REST_V1}/pods"),
    )
}

pub fn dry_run_schedule_pod(
    specification: &PodCreateSpec,
    terminate_after: &str,
    api_key_environment: &str,
) -> Result<Value> {
    validate_environment_name(api_key_environment)?;
    let request = build_graphql_pod_request(specification, terminate_after)?;
    Ok(json!({
        "method": "POST",
        "url": RUNPOD_GRAPHQL,
        "headers": {
            "Authorization": REDACTED_BEARER,
            "Content-Type": "application/json"
        },
        "body": request,
        "termination_binding": "requested_unobservable",
        "environment": {
            "control_plane_api_key": {
                "source": api_key_environment,
                "value": "<read-at-runtime-and-never-sent-to-pod>"
            },
            "pod_env": "no environment variables are exposed"
        }
    }))
}

fn dry_run_create_pod_at(
    specification: &PodCreateSpec,
    api_key_environment: &str,
    url: &str,
) -> Result<Value> {
    let request = build_pod_request(specification)?;
    Ok(json!({
        "method": "POST",
        "url": url,
        "headers": {
            "Authorization": REDACTED_BEARER,
            "Content-Type": "application/json"
        },
        "body": request,
        "local_policy": {
            "maximum_adjusted_hourly_price": specification.maximum_adjusted_hourly_price,
            "worker_binary": specification.worker_binary
        },
        "environment": {
            "control_plane_api_key": {
                "source": api_key_environment,
                "value": "<read-at-runtime-and-never-sent-to-pod>"
            },
            "pod_env": "no environment variables are exposed"
        }
    }))
}

impl RunpodClient {
    /// Ask RunPod's scheduler to terminate the Pod at an absolute UTC time.
    /// The GraphQL schema accepts this field but does not expose it for read-back,
    /// so the returned receipt records it as requested rather than observed.
    pub async fn schedule_pod_with_termination(
        &self,
        specification: &PodCreateSpec,
        terminate_after: &str,
    ) -> Result<ScheduledPod> {
        let request = build_graphql_pod_request(specification, terminate_after)?;
        let request_bytes = rag_pipeline::canonical_json_bytes(&request)
            .map_err(|_| RunpodControlError::Invalid("GraphQL create request"))?;
        let envelope: GraphqlCreateEnvelope = self
            .send_json(
                "schedule Pod with termination",
                self.authorized(self.http.post(&self.graphql_url))
                    .json(&request),
                StatusCode::OK,
            )
            .await?;
        let id = envelope
            .data
            .and_then(|data| data.pod_find_and_deploy_on_demand)
            .map(|pod| pod.id)
            .ok_or_else(|| RunpodControlError::SchedulerRejected {
                reason: sanitize_graphql_rejection(&envelope.errors),
            })?;
        validate_identifier(&id).map_err(|_| RunpodControlError::MalformedResponse {
            operation: "schedule Pod with termination",
        })?;
        Ok(ScheduledPod {
            schema_version: "livefire.rag.runpod-scheduled-pod/1",
            pod_id: id,
            terminate_after: terminate_after.to_owned(),
            termination_binding: "requested_unobservable",
            graphql_request_sha256: rag_pipeline::digest_bytes(&request_bytes),
            graphql_request: serde_json::to_value(request)
                .map_err(|_| RunpodControlError::Invalid("GraphQL create request"))?,
        })
    }

    pub async fn admit_scheduled_pod(
        &self,
        scheduled: &ScheduledPod,
        specification: &PodCreateSpec,
    ) -> Result<Pod> {
        let started = std::time::Instant::now();
        loop {
            match self.get_pod_value(&scheduled.pod_id).await {
                Ok((status, value)) => {
                    if status == PodDesiredStatus::Created {
                        // Scheduling responses intentionally omit the complete
                        // machine, GPU, price, and network identity. Require
                        // those only after the provider reports RUNNING.
                    } else {
                        let pod = match pod_from_value(value, &scheduled.pod_id) {
                            Ok(pod) => pod,
                            Err(error) => {
                                let _ = self.delete_pod(&scheduled.pod_id).await;
                                return Err(error);
                            }
                        };
                        if let Some(reason) = pod.admission_rejection(specification) {
                            let cleanup = if self.delete_pod(&pod.id).await.is_ok() {
                                CleanupOutcome::Succeeded
                            } else {
                                CleanupOutcome::Failed
                            };
                            return Err(RunpodControlError::CreatedPodRejected { reason, cleanup });
                        }
                        if pod.desired_status == PodDesiredStatus::Running {
                            return Ok(pod);
                        }
                    }
                }
                Err(RunpodControlError::UnexpectedStatus { status: 404, .. }) => {}
                Err(error) => {
                    let _ = self.delete_pod(&scheduled.pod_id).await;
                    return Err(error);
                }
            }
            if started.elapsed() >= Duration::from_secs(180) {
                let cleanup = if self.delete_pod(&scheduled.pod_id).await.is_ok() {
                    CleanupOutcome::Succeeded
                } else {
                    CleanupOutcome::Failed
                };
                return Err(RunpodControlError::CreatedPodRejected {
                    reason: "Pod did not reach running status before the admission deadline",
                    cleanup,
                });
            }
            tokio::time::sleep(admission_poll_interval()).await;
        }
    }

    pub async fn create_network_volume(
        &self,
        specification: &NetworkVolumeCreateSpec,
    ) -> Result<NetworkVolume> {
        specification.validate()?;
        let request = NetworkVolumeCreateRequest::from(specification);
        let volume: NetworkVolume = self
            .send_json(
                "create network volume",
                self.authorized(self.http.post(self.url("networkvolumes")))
                    .json(&request),
                StatusCode::CREATED,
            )
            .await?;
        volume
            .validate()
            .map_err(|_| RunpodControlError::MalformedResponse {
                operation: "create network volume",
            })?;
        if volume.name != specification.name
            || volume.size != specification.size
            || volume.data_center_id != specification.data_center_id
        {
            return Err(RunpodControlError::MalformedResponse {
                operation: "create network volume",
            });
        }
        Ok(volume)
    }

    pub async fn get_network_volume(&self, id: &str) -> Result<NetworkVolume> {
        validate_identifier(id)?;
        let volume: NetworkVolume = self
            .send_json(
                "get network volume",
                self.authorized(self.http.get(self.url(&format!("networkvolumes/{id}")))),
                StatusCode::OK,
            )
            .await?;
        volume
            .validate()
            .map_err(|_| RunpodControlError::MalformedResponse {
                operation: "get network volume",
            })?;
        if volume.id != id {
            return Err(RunpodControlError::MalformedResponse {
                operation: "get network volume",
            });
        }
        Ok(volume)
    }

    pub async fn delete_network_volume(&self, id: &str) -> Result<()> {
        validate_identifier(id)?;
        self.send_empty(
            "delete network volume",
            self.authorized(self.http.delete(self.url(&format!("networkvolumes/{id}")))),
            StatusCode::NO_CONTENT,
        )
        .await
    }

    #[allow(dead_code)] // Retained for REST adapter compatibility tests; paid launches use GraphQL TTL.
    pub async fn create_pod(&self, specification: &PodCreateSpec) -> Result<Pod> {
        let request = build_pod_request(specification)?;
        let pod: Pod = self
            .send_json(
                "create Pod",
                self.authorized(self.http.post(self.url("pods")))
                    .json(&request),
                StatusCode::CREATED,
            )
            .await?;
        let rejection = pod.admission_rejection(specification);
        if let Some(reason) = rejection {
            let cleanup =
                if validate_identifier(&pod.id).is_ok() && self.delete_pod(&pod.id).await.is_ok() {
                    CleanupOutcome::Succeeded
                } else {
                    CleanupOutcome::Failed
                };
            return Err(RunpodControlError::CreatedPodRejected { reason, cleanup });
        }
        Ok(pod)
    }

    pub async fn get_pod(&self, id: &str) -> Result<Pod> {
        let (_, value) = self.get_pod_value(id).await?;
        pod_from_value(value, id)
    }

    async fn get_pod_value(&self, id: &str) -> Result<(PodDesiredStatus, Value)> {
        validate_identifier(id)?;
        let value: Value = self
            .send_json(
                "get Pod",
                self.authorized(self.http.get(self.url(&format!(
                    "pods/{id}?includeMachine=true&includeNetworkVolume=true"
                )))),
                StatusCode::OK,
            )
            .await?;
        let status: PodStatusResponse = serde_json::from_value(value.clone()).map_err(|error| {
            let reason: String = error
                .to_string()
                .chars()
                .filter(|character| !character.is_control())
                .take(256)
                .collect();
            RunpodControlError::MalformedJsonResponse {
                operation: "get Pod status",
                reason,
            }
        })?;
        if status.id != id {
            return Err(RunpodControlError::MalformedResponse {
                operation: "get Pod status",
            });
        }
        Ok((status.desired_status, value))
    }

    pub async fn delete_pod(&self, id: &str) -> Result<()> {
        validate_identifier(id)?;
        self.send_empty(
            "delete Pod",
            self.authorized(self.http.delete(self.url(&format!("pods/{id}")))),
            StatusCode::NO_CONTENT,
        )
        .await
    }

    fn authorized(&self, request: RequestBuilder) -> RequestBuilder {
        request.header(AUTHORIZATION, self.authorization.clone())
    }

    fn url(&self, suffix: &str) -> String {
        format!("{}/{suffix}", self.base_url)
    }

    async fn send_json<T: DeserializeOwned>(
        &self,
        operation: &'static str,
        request: RequestBuilder,
        expected: StatusCode,
    ) -> Result<T> {
        let response = request
            .send()
            .await
            .map_err(|_| RunpodControlError::Transport { operation })?;
        let status = response.status();
        if status != expected {
            return Err(RunpodControlError::UnexpectedStatus {
                operation,
                status: status.as_u16(),
            });
        }
        let bytes = bounded_body(response, operation, self.maximum_response_bytes).await?;
        let value: Value = serde_json::from_slice(&bytes).map_err(|error| {
            let reason: String = error
                .to_string()
                .chars()
                .filter(|character| !character.is_control())
                .take(256)
                .collect();
            RunpodControlError::MalformedJsonResponse { operation, reason }
        })?;
        let shape = match &value {
            Value::Object(object) => format!(
                "top-level object keys [{}]",
                object.keys().cloned().collect::<Vec<_>>().join(",")
            ),
            Value::Array(array) => format!("top-level array length {}", array.len()),
            Value::Null => "top-level null".to_owned(),
            Value::Bool(_) => "top-level boolean".to_owned(),
            Value::Number(_) => "top-level number".to_owned(),
            Value::String(_) => "top-level string".to_owned(),
        };
        serde_json::from_value(value).map_err(|error| {
            let reason: String = format!("{error}; {shape}")
                .chars()
                .filter(|character| !character.is_control())
                .take(512)
                .collect();
            RunpodControlError::MalformedJsonResponse { operation, reason }
        })
    }

    async fn send_empty(
        &self,
        operation: &'static str,
        request: RequestBuilder,
        expected: StatusCode,
    ) -> Result<()> {
        let response = request
            .send()
            .await
            .map_err(|_| RunpodControlError::Transport { operation })?;
        let status = response.status();
        if status != expected {
            return Err(RunpodControlError::UnexpectedStatus {
                operation,
                status: status.as_u16(),
            });
        }
        let body = bounded_body(response, operation, self.maximum_response_bytes).await?;
        if !body.is_empty() {
            return Err(RunpodControlError::MalformedResponse { operation });
        }
        Ok(())
    }
}

fn admission_poll_interval() -> Duration {
    if cfg!(test) {
        Duration::from_millis(1)
    } else {
        Duration::from_secs(2)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduledPod {
    pub schema_version: &'static str,
    pub pod_id: String,
    pub terminate_after: String,
    pub termination_binding: &'static str,
    pub graphql_request_sha256: rag_pipeline::Digest,
    pub graphql_request: Value,
}

#[derive(Debug, Serialize)]
struct GraphqlCreateRequest<'a> {
    query: &'static str,
    variables: GraphqlCreateVariables<'a>,
}

#[derive(Debug, Serialize)]
struct GraphqlCreateVariables<'a> {
    input: GraphqlCreateInput<'a>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GraphqlCreateInput<'a> {
    cloud_type: &'static str,
    container_disk_in_gb: u32,
    data_center_id: &'a str,
    gpu_count: u8,
    gpu_type_id: &'a str,
    allowed_cuda_versions: [&'a str; 1],
    image_name: &'a str,
    name: &'a str,
    ports: &'static str,
    start_ssh: bool,
    support_public_ip: bool,
    volume_mount_path: &'static str,
    network_volume_id: &'a str,
    docker_args: String,
    terminate_after: &'a str,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphqlCreateEnvelope {
    data: Option<GraphqlCreateData>,
    #[serde(default)]
    errors: Vec<GraphqlError>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphqlCreateData {
    #[serde(rename = "podFindAndDeployOnDemand")]
    pod_find_and_deploy_on_demand: Option<GraphqlCreatedPod>,
}

#[derive(Debug, Deserialize)]
struct GraphqlError {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphqlCreatedPod {
    id: String,
}

fn sanitize_graphql_rejection(errors: &[GraphqlError]) -> String {
    let reason = errors
        .first()
        .map(|error| error.message.as_str())
        .unwrap_or("request was rejected without a reason");
    let sanitized: String = reason
        .chars()
        .filter(|character| !character.is_control())
        .take(256)
        .collect();
    if sanitized.is_empty() {
        "request was rejected without a reason".into()
    } else {
        sanitized
    }
}

fn build_graphql_pod_request<'a>(
    specification: &'a PodCreateSpec,
    terminate_after: &'a str,
) -> Result<GraphqlCreateRequest<'a>> {
    specification.validate()?;
    let parsed = chrono::DateTime::parse_from_rfc3339(terminate_after)
        .map_err(|_| RunpodControlError::Invalid("Pod termination deadline"))?;
    if parsed.offset().local_minus_utc() != 0
        || specification.worker_arguments.iter().any(|argument| {
            !argument.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'/' | b':' | b'-')
            })
        })
    {
        return Err(RunpodControlError::Invalid(
            "Pod termination deadline or GraphQL worker arguments",
        ));
    }
    Ok(GraphqlCreateRequest {
        query: "mutation createPod($input: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: $input) { id } }",
        variables: GraphqlCreateVariables {
            input: GraphqlCreateInput {
                cloud_type: "SECURE",
                container_disk_in_gb: specification.container_disk_gb,
                data_center_id: &specification.network_volume.data_center_id,
                gpu_count: 1,
                gpu_type_id: &specification.gpu_type_id,
                allowed_cuda_versions: [specification.allowed_cuda_version.as_str()],
                image_name: &specification.image,
                name: &specification.name,
                ports: "",
                start_ssh: false,
                support_public_ip: false,
                volume_mount_path: WORKSPACE_MOUNT,
                network_volume_id: &specification.network_volume.id,
                docker_args: specification.worker_arguments.join(" "),
                terminate_after,
            },
        },
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkVolumeCreateSpec {
    pub name: String,
    pub size: u32,
    pub data_center_id: String,
}

impl NetworkVolumeCreateSpec {
    fn validate(&self) -> Result<()> {
        validate_name(&self.name)?;
        validate_identifier(&self.data_center_id)?;
        if !(1..=4000).contains(&self.size) {
            return Err(RunpodControlError::Invalid("network volume size"));
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct NetworkVolumeCreateRequest<'a> {
    data_center_id: &'a str,
    name: &'a str,
    size: u32,
}

impl<'a> From<&'a NetworkVolumeCreateSpec> for NetworkVolumeCreateRequest<'a> {
    fn from(value: &'a NetworkVolumeCreateSpec) -> Self {
        Self {
            data_center_id: &value.data_center_id,
            name: &value.name,
            size: value.size,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NetworkVolume {
    pub id: String,
    pub name: String,
    pub size: u32,
    pub data_center_id: String,
}

impl NetworkVolume {
    fn validate(&self) -> Result<()> {
        validate_identifier(&self.id)?;
        validate_identifier(&self.data_center_id)?;
        validate_name(&self.name)?;
        if !(1..=4000).contains(&self.size) {
            return Err(RunpodControlError::MalformedResponse {
                operation: "network volume",
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum RunpodCudaVersion {
    #[value(name = "11.8")]
    Cuda11_8,
    #[value(name = "12.0")]
    Cuda12_0,
    #[value(name = "12.1")]
    Cuda12_1,
    #[value(name = "12.2")]
    Cuda12_2,
    #[value(name = "12.3")]
    Cuda12_3,
    #[value(name = "12.4")]
    Cuda12_4,
    #[value(name = "12.5")]
    Cuda12_5,
    #[value(name = "12.6")]
    Cuda12_6,
    #[value(name = "12.7")]
    Cuda12_7,
    #[value(name = "12.8")]
    Cuda12_8,
    #[value(name = "12.9")]
    Cuda12_9,
    #[value(name = "13.0")]
    Cuda13_0,
}

impl RunpodCudaVersion {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cuda11_8 => "11.8",
            Self::Cuda12_0 => "12.0",
            Self::Cuda12_1 => "12.1",
            Self::Cuda12_2 => "12.2",
            Self::Cuda12_3 => "12.3",
            Self::Cuda12_4 => "12.4",
            Self::Cuda12_5 => "12.5",
            Self::Cuda12_6 => "12.6",
            Self::Cuda12_7 => "12.7",
            Self::Cuda12_8 => "12.8",
            Self::Cuda12_9 => "12.9",
            Self::Cuda13_0 => "13.0",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PodCreateSpec {
    pub name: String,
    pub image: String,
    pub gpu_type_id: String,
    pub allowed_cuda_version: RunpodCudaVersion,
    pub network_volume: NetworkVolume,
    pub worker_binary: rag_pipeline::ComponentRef,
    pub worker_arguments: Vec<String>,
    pub container_disk_gb: u32,
    pub maximum_adjusted_hourly_price: f64,
}

impl PodCreateSpec {
    fn validate(&self) -> Result<()> {
        validate_name(&self.name)?;
        validate_pinned_image(&self.image)?;
        validate_name(&self.gpu_type_id)?;
        self.network_volume.validate()?;
        self.worker_binary
            .validate()
            .map_err(|_| RunpodControlError::Invalid("worker binary identity"))?;
        if self.worker_arguments.len() > 64
            || self.worker_arguments.iter().any(|value| {
                value.is_empty()
                    || value.len() > 4096
                    || value.contains('\0')
                    || value.contains("Bearer ")
            })
            || !(1..=200).contains(&self.container_disk_gb)
            || !self.maximum_adjusted_hourly_price.is_finite()
            || self.maximum_adjusted_hourly_price <= 0.0
            || self.maximum_adjusted_hourly_price > 100.0
        {
            return Err(RunpodControlError::Invalid("Pod create specification"));
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PodCreateRequest<'a> {
    name: &'a str,
    image_name: &'a str,
    cloud_type: &'static str,
    compute_type: &'static str,
    gpu_type_ids: [&'a str; 1],
    allowed_cuda_versions: [&'a str; 1],
    gpu_type_priority: &'static str,
    gpu_count: u8,
    data_center_ids: [&'a str; 1],
    data_center_priority: &'static str,
    network_volume_id: &'a str,
    volume_mount_path: &'static str,
    container_disk_in_gb: u32,
    interruptible: bool,
    locked: bool,
    global_networking: bool,
    ports: [String; 0],
    docker_entrypoint: [&'static str; 1],
    docker_start_cmd: &'a [String],
    env: BTreeMap<String, String>,
}

fn build_pod_request(specification: &PodCreateSpec) -> Result<PodCreateRequest<'_>> {
    specification.validate()?;
    Ok(PodCreateRequest {
        name: &specification.name,
        image_name: &specification.image,
        cloud_type: "SECURE",
        compute_type: "GPU",
        gpu_type_ids: [&specification.gpu_type_id],
        allowed_cuda_versions: [specification.allowed_cuda_version.as_str()],
        gpu_type_priority: "custom",
        gpu_count: 1,
        data_center_ids: [&specification.network_volume.data_center_id],
        data_center_priority: "custom",
        network_volume_id: &specification.network_volume.id,
        volume_mount_path: WORKSPACE_MOUNT,
        container_disk_in_gb: specification.container_disk_gb,
        interruptible: false,
        locked: false,
        global_networking: false,
        ports: [],
        docker_entrypoint: [WORKER_IMAGE_PATH],
        docker_start_cmd: &specification.worker_arguments,
        env: BTreeMap::new(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PodDesiredStatus {
    #[serde(rename = "CREATED")]
    Created,
    #[serde(rename = "RUNNING")]
    Running,
    #[serde(rename = "EXITED")]
    Exited,
    #[serde(rename = "TERMINATED")]
    Terminated,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PodStatusResponse {
    id: String,
    desired_status: PodDesiredStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PodGpu {
    pub id: String,
    pub count: u32,
    pub display_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PodMachine {
    pub id: String,
    pub secure_cloud: bool,
    pub data_center_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Pod {
    pub id: String,
    pub name: String,
    pub image: String,
    pub adjusted_cost_per_hr: f64,
    pub desired_status: PodDesiredStatus,
    pub interruptible: bool,
    pub gpu: PodGpu,
    pub machine: PodMachine,
    pub network_volume: NetworkVolume,
    pub ports: Vec<String>,
    pub port_mappings: BTreeMap<String, u16>,
    pub volume_mount_path: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RestPodResponse {
    id: String,
    name: String,
    image_name: String,
    #[serde(default)]
    adjusted_cost_per_hr: Option<Value>,
    cost_per_hr: Value,
    desired_status: PodDesiredStatus,
    #[serde(default)]
    interruptible: bool,
    gpu_count: u32,
    machine_id: String,
    machine: RestPodMachine,
    network_volume: NetworkVolume,
    #[serde(default)]
    ports: Vec<String>,
    #[serde(default)]
    port_mappings: BTreeMap<String, u16>,
    volume_mount_path: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RestPodMachine {
    secure_cloud: bool,
    data_center_id: String,
    gpu_type_id: String,
    #[serde(default)]
    gpu_display_name: Option<String>,
}

fn json_price(value: &Value) -> Option<f64> {
    match value {
        Value::Number(value) => value.as_f64(),
        Value::String(value) => value.parse().ok(),
        _ => None,
    }
    .filter(|value| value.is_finite() && *value >= 0.0)
}

fn rest_pod_from_value(value: Value) -> Result<Pod> {
    let response: RestPodResponse = serde_json::from_value(value).map_err(|error| {
        let reason: String = error
            .to_string()
            .chars()
            .filter(|character| !character.is_control())
            .take(256)
            .collect();
        RunpodControlError::MalformedJsonResponse {
            operation: "get current REST Pod",
            reason,
        }
    })?;
    let adjusted_cost_per_hr = response
        .adjusted_cost_per_hr
        .as_ref()
        .and_then(json_price)
        .or_else(|| json_price(&response.cost_per_hr))
        .ok_or(RunpodControlError::MalformedResponse {
            operation: "get current REST Pod price",
        })?;
    let display_name = response
        .machine
        .gpu_display_name
        .unwrap_or_else(|| response.machine.gpu_type_id.clone());
    Ok(Pod {
        id: response.id,
        name: response.name,
        image: response.image_name,
        adjusted_cost_per_hr,
        desired_status: response.desired_status,
        // podFindAndDeployOnDemand is RunPod's non-spot scheduling operation.
        // A returned interruptible field, if present, overrides this default.
        interruptible: response.interruptible,
        gpu: PodGpu {
            id: response.machine.gpu_type_id,
            count: response.gpu_count,
            display_name,
        },
        machine: PodMachine {
            id: response.machine_id,
            secure_cloud: response.machine.secure_cloud,
            data_center_id: response.machine.data_center_id,
        },
        network_volume: response.network_volume,
        ports: response.ports,
        port_mappings: response.port_mappings,
        volume_mount_path: response.volume_mount_path,
    })
}

fn pod_from_value(value: Value, expected_id: &str) -> Result<Pod> {
    let shape = match &value {
        Value::Object(object) => {
            let top = object.keys().cloned().collect::<Vec<_>>().join(",");
            let nested = ["gpu", "machine", "networkVolume"]
                .into_iter()
                .filter_map(|name| {
                    object.get(name).and_then(Value::as_object).map(|child| {
                        format!(
                            "{name}=[{}]",
                            child.keys().cloned().collect::<Vec<_>>().join(",")
                        )
                    })
                })
                .collect::<Vec<_>>()
                .join(";");
            format!("top=[{top}]; nested {nested}")
        }
        _ => "top-level value is not an object".to_owned(),
    };
    let pod: Pod = match serde_json::from_value(value.clone()) {
        Ok(pod) => pod,
        Err(first_error) => rest_pod_from_value(value).map_err(|second_error| {
            let reason: String = format!("{first_error}; {second_error}; {shape}")
                .chars()
                .filter(|character| !character.is_control())
                .take(1024)
                .collect();
            RunpodControlError::MalformedJsonResponse {
                operation: "get Pod",
                reason,
            }
        })?,
    };
    pod.validate()
        .map_err(|_| RunpodControlError::MalformedResponse {
            operation: "get Pod",
        })?;
    if pod.id != expected_id {
        return Err(RunpodControlError::MalformedResponse {
            operation: "get Pod",
        });
    }
    Ok(pod)
}

impl Pod {
    fn validate(&self) -> Result<()> {
        validate_identifier(&self.id)?;
        validate_name(&self.name)?;
        validate_pinned_image(&self.image)?;
        validate_name(&self.gpu.id)?;
        validate_name(&self.gpu.display_name)?;
        validate_identifier(&self.machine.id)?;
        validate_identifier(&self.machine.data_center_id)?;
        self.network_volume.validate()?;
        if !self.adjusted_cost_per_hr.is_finite() || self.adjusted_cost_per_hr < 0.0 {
            return Err(RunpodControlError::MalformedResponse { operation: "Pod" });
        }
        Ok(())
    }

    fn admission_rejection(&self, requested: &PodCreateSpec) -> Option<&'static str> {
        if self.validate().is_err() {
            return Some("malformed Pod identity");
        }
        if self.adjusted_cost_per_hr > requested.maximum_adjusted_hourly_price {
            return Some("adjusted hourly price exceeds the explicit maximum");
        }
        if self.name != requested.name || self.desired_status != PodDesiredStatus::Running {
            return Some("Pod name or initial status differs from the request");
        }
        if !self.machine.secure_cloud {
            return Some("Pod is not in Secure Cloud");
        }
        if self.gpu.count != 1 || self.gpu.id != requested.gpu_type_id {
            return Some("GPU identity or count differs from the request");
        }
        if self.network_volume.id != requested.network_volume.id
            || self.network_volume.data_center_id != requested.network_volume.data_center_id
            || self.machine.data_center_id != requested.network_volume.data_center_id
            || self.volume_mount_path != WORKSPACE_MOUNT
        {
            return Some("network volume identity, location, or mount differs from the request");
        }
        if self.image != requested.image {
            return Some("container image differs from the pinned request");
        }
        if self.interruptible {
            return Some("Pod is interruptible");
        }
        if !self.ports.is_empty() || !self.port_mappings.is_empty() {
            return Some("Pod exposes a network port");
        }
        None
    }
}

async fn bounded_body(
    mut response: reqwest::Response,
    operation: &'static str,
    limit: usize,
) -> Result<Vec<u8>> {
    if response
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .is_some_and(|length| length > limit as u64)
    {
        return Err(RunpodControlError::ResponseTooLarge { operation, limit });
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| RunpodControlError::Transport { operation })?
    {
        if body
            .len()
            .checked_add(chunk.len())
            .is_none_or(|length| length > limit)
        {
            return Err(RunpodControlError::ResponseTooLarge { operation, limit });
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn validate_limits(limits: RunpodClientLimits) -> Result<()> {
    if limits.timeout.is_zero()
        || limits.timeout > Duration::from_secs(120)
        || limits.maximum_response_bytes == 0
        || limits.maximum_response_bytes > MAX_RESPONSE_BYTES
    {
        return Err(RunpodControlError::Invalid("HTTP limits"));
    }
    Ok(())
}

fn validate_environment_name(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_')
        || value.as_bytes()[0].is_ascii_digit()
    {
        return Err(RunpodControlError::Invalid("environment variable name"));
    }
    Ok(())
}

fn validate_identifier(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 191
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(RunpodControlError::Invalid("RunPod identifier"));
    }
    Ok(())
}

fn validate_name(value: &str) -> Result<()> {
    if value.trim().is_empty() || value.len() > 191 || value.chars().any(char::is_control) {
        return Err(RunpodControlError::Invalid("RunPod resource name"));
    }
    Ok(())
}

fn validate_pinned_image(value: &str) -> Result<()> {
    let Some((name, digest)) = value.split_once("@sha256:") else {
        return Err(RunpodControlError::Invalid("pinned container image"));
    };
    if name.is_empty()
        || digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(RunpodControlError::Invalid("pinned container image"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Read, Write},
        net::{TcpListener, TcpStream},
        sync::mpsc,
        thread,
    };

    const SECRET: &str = "runpod-secret-key-material";
    const IMAGE: &str = "ghcr.io/huggingface/text-embeddings-inference@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    #[derive(Clone)]
    struct FakeResponse {
        status: u16,
        body: Vec<u8>,
        declared_length: Option<usize>,
    }

    fn response(status: u16, body: impl Into<Vec<u8>>) -> FakeResponse {
        FakeResponse {
            status,
            body: body.into(),
            declared_length: None,
        }
    }

    fn spawn_server(
        responses: Vec<FakeResponse>,
    ) -> (String, mpsc::Receiver<String>, thread::JoinHandle<()>) {
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
                    201 => "Created",
                    204 => "No Content",
                    400 => "Bad Request",
                    500 => "Internal Server Error",
                    _ => "Status",
                };
                let length = response.declared_length.unwrap_or(response.body.len());
                write!(
                    stream,
                    "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    response.status, reason, length
                )
                .unwrap();
                stream.write_all(&response.body).unwrap();
            }
        });
        (format!("http://{address}/v1"), receiver, handle)
    }

    fn read_request(stream: &mut TcpStream) -> String {
        let mut bytes = Vec::new();
        let mut buffer = [0_u8; 4096];
        let header_end = loop {
            let count = stream.read(&mut buffer).unwrap();
            assert!(count > 0);
            bytes.extend_from_slice(&buffer[..count]);
            if let Some(offset) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                break offset + 4;
            }
        };
        let header = String::from_utf8_lossy(&bytes[..header_end]);
        let content_length = header
            .lines()
            .find_map(|line| {
                line.to_ascii_lowercase()
                    .strip_prefix("content-length: ")
                    .and_then(|value| value.parse::<usize>().ok())
            })
            .unwrap_or(0);
        while bytes.len() < header_end + content_length {
            let count = stream.read(&mut buffer).unwrap();
            assert!(count > 0);
            bytes.extend_from_slice(&buffer[..count]);
        }
        String::from_utf8(bytes).unwrap()
    }

    fn client(base_url: &str, maximum_response_bytes: usize) -> RunpodClient {
        RunpodClient::from_api_key_at(
            "RUNPOD_API_KEY",
            SECRET,
            RunpodClientLimits {
                timeout: Duration::from_secs(2),
                maximum_response_bytes,
            },
            base_url,
        )
        .unwrap()
    }

    fn volume() -> NetworkVolume {
        NetworkVolume {
            id: "volume-1".into(),
            name: "rag-volume".into(),
            size: 100,
            data_center_id: "US-KS-2".into(),
        }
    }

    fn pod_spec() -> PodCreateSpec {
        PodCreateSpec {
            name: "rag-worker".into(),
            image: IMAGE.into(),
            gpu_type_id: "NVIDIA H100 80GB HBM3".into(),
            allowed_cuda_version: RunpodCudaVersion::Cuda12_9,
            network_volume: volume(),
            worker_binary: rag_pipeline::ComponentRef {
                id: "livefire.rag.runpod-worker".into(),
                version: "1".into(),
                sha256: "b".repeat(64).parse().unwrap(),
            },
            worker_arguments: vec!["--bundle".into(), "/workspace/run/bundle.json".into()],
            container_disk_gb: 50,
            maximum_adjusted_hourly_price: 1.0,
        }
    }

    fn pod_json(price: f64) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "id":"pod-1", "name":"rag-worker", "image":IMAGE,
            "adjustedCostPerHr":price, "desiredStatus":"RUNNING", "interruptible":false,
            "gpu":{"id":"NVIDIA H100 80GB HBM3","count":1,"displayName":"NVIDIA H100 80GB HBM3"},
            "machine":{"id":"machine-1","secureCloud":true,"dataCenterId":"US-KS-2"},
            "networkVolume":volume(), "ports":[], "portMappings":{}, "volumeMountPath":"/workspace"
        }))
        .unwrap()
    }

    fn current_rest_pod_json(price: f64) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "id":"pod-1", "name":"rag-worker", "imageName":IMAGE,
            "costPerHr":price, "desiredStatus":"RUNNING", "gpuCount":1,
            "machineId":"machine-1",
            "machine":{
                "secureCloud":true, "dataCenterId":"US-KS-2",
                "gpuTypeId":"NVIDIA H100 80GB HBM3",
                "location":"test", "supportPublicIp":true
            },
            "networkVolume":volume(), "networkVolumeId":"volume-1",
            "volumeMountPath":"/workspace", "publicIp":"192.0.2.1"
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn current_rest_pod_shape_is_adapted_without_weakening_identity() {
        let (base, _, server) = spawn_server(vec![response(200, current_rest_pod_json(0.75))]);
        let pod = client(&base, 4096).get_pod("pod-1").await.unwrap();
        assert_eq!(pod.id, "pod-1");
        assert_eq!(pod.image, IMAGE);
        assert_eq!(pod.adjusted_cost_per_hr, 0.75);
        assert_eq!(pod.gpu.id, "NVIDIA H100 80GB HBM3");
        assert_eq!(pod.gpu.count, 1);
        assert_eq!(pod.machine.id, "machine-1");
        assert!(pod.machine.secure_cloud);
        assert_eq!(pod.network_volume, volume());
        assert!(pod.ports.is_empty());
        assert!(pod.port_mappings.is_empty());
        server.join().unwrap();
    }

    #[tokio::test]
    async fn volume_create_get_delete_uses_bearer_auth_and_exact_shapes() {
        let volume_body = serde_json::to_vec(&volume()).unwrap();
        let (base, requests, server) = spawn_server(vec![
            response(201, volume_body.clone()),
            response(200, volume_body),
            response(204, Vec::new()),
        ]);
        let client = client(&base, 4096);
        let specification = NetworkVolumeCreateSpec {
            name: "rag-volume".into(),
            size: 100,
            data_center_id: "US-KS-2".into(),
        };
        assert_eq!(
            client.create_network_volume(&specification).await.unwrap(),
            volume()
        );
        assert_eq!(
            client.get_network_volume("volume-1").await.unwrap(),
            volume()
        );
        client.delete_network_volume("volume-1").await.unwrap();
        for expected in [
            "POST /v1/networkvolumes ",
            "GET /v1/networkvolumes/volume-1 ",
            "DELETE /v1/networkvolumes/volume-1 ",
        ] {
            let request = requests.recv().unwrap();
            assert!(request.starts_with(expected));
            assert!(request.contains(&format!("authorization: Bearer {SECRET}")));
        }
        server.join().unwrap();
    }

    #[tokio::test]
    async fn pod_create_status_delete_is_private_secure_and_fixed_to_one_gpu() {
        let (base, requests, server) = spawn_server(vec![
            response(201, pod_json(0.75)),
            response(200, pod_json(0.75)),
            response(204, Vec::new()),
        ]);
        let client = client(&base, 8192);
        let pod = client.create_pod(&pod_spec()).await.unwrap();
        assert_eq!(pod.desired_status, PodDesiredStatus::Running);
        client.get_pod("pod-1").await.unwrap();
        client.delete_pod("pod-1").await.unwrap();

        let create = requests.recv().unwrap();
        let body = create.split("\r\n\r\n").nth(1).unwrap();
        let body: Value = serde_json::from_str(body).unwrap();
        assert_eq!(body["cloudType"], "SECURE");
        assert_eq!(body["computeType"], "GPU");
        assert_eq!(body["gpuCount"], 1);
        assert_eq!(body["allowedCudaVersions"], json!(["12.9"]));
        assert_eq!(body["interruptible"], false);
        assert_eq!(body["ports"], json!([]));
        assert_eq!(body["env"], json!({}));
        assert_eq!(body["volumeMountPath"], WORKSPACE_MOUNT);
        assert_eq!(body["dockerEntrypoint"], json!([WORKER_IMAGE_PATH]));
        assert!(requests.recv().unwrap().starts_with("GET /v1/pods/pod-1?"));
        assert!(
            requests
                .recv()
                .unwrap()
                .starts_with("DELETE /v1/pods/pod-1 ")
        );
        server.join().unwrap();
    }

    #[tokio::test]
    async fn graphql_schedule_sends_absolute_termination_then_admits_through_rest() {
        let scheduled_body = serde_json::to_vec(&json!({
            "data":{"podFindAndDeployOnDemand":{"id":"pod-1"}}
        }))
        .unwrap();
        let (base, requests, server) = spawn_server(vec![
            response(200, scheduled_body),
            response(200, pod_json(0.75)),
            response(204, Vec::new()),
        ]);
        let client = client(&base, 8192);
        let deadline = "2026-08-18T00:00:00Z";
        let scheduled = client
            .schedule_pod_with_termination(&pod_spec(), deadline)
            .await
            .unwrap();
        assert_eq!(scheduled.pod_id, "pod-1");
        assert_eq!(scheduled.terminate_after, deadline);
        assert_eq!(scheduled.termination_binding, "requested_unobservable");
        let pod = client
            .admit_scheduled_pod(&scheduled, &pod_spec())
            .await
            .unwrap();
        assert_eq!(pod.id, "pod-1");
        client.delete_pod(&pod.id).await.unwrap();

        let create = requests.recv().unwrap();
        assert!(create.starts_with("POST /graphql "));
        let body_text = create.split("\r\n\r\n").nth(1).unwrap();
        let body: Value = serde_json::from_str(body_text).unwrap();
        assert_eq!(
            body.pointer("/variables/input/terminateAfter")
                .and_then(Value::as_str),
            Some(deadline)
        );
        assert_eq!(
            body.pointer("/variables/input/dockerArgs")
                .and_then(Value::as_str),
            Some("--bundle /workspace/run/bundle.json")
        );
        assert_eq!(
            body.pointer("/variables/input/allowedCudaVersions"),
            Some(&json!(["12.9"]))
        );
        assert!(!body_text.contains(SECRET));
        assert!(requests.recv().unwrap().starts_with("GET /v1/pods/pod-1"));
        assert!(
            requests
                .recv()
                .unwrap()
                .starts_with("DELETE /v1/pods/pod-1")
        );
        server.join().unwrap();
    }

    #[tokio::test]
    async fn graphql_scheduler_rejection_reports_only_a_bounded_reason() {
        let rejected = serde_json::to_vec(&json!({
            "data":{"podFindAndDeployOnDemand":null},
            "errors":[{
                "message":"No compatible CUDA host is currently available.\n",
                "path":["podFindAndDeployOnDemand"]
            }]
        }))
        .unwrap();
        let (base, _, server) = spawn_server(vec![response(200, rejected)]);
        let error = client(&base, 8192)
            .schedule_pod_with_termination(&pod_spec(), "2026-08-18T00:00:00Z")
            .await
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "RunPod scheduler rejected the Pod request: No compatible CUDA host is currently available."
        );
        assert!(!error.to_string().contains(SECRET));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn scheduled_pod_admission_retries_not_found_and_created() {
        let mut created: Value = serde_json::from_slice(&pod_json(0.75)).unwrap();
        created["desiredStatus"] = Value::String("CREATED".into());
        let (base, requests, server) = spawn_server(vec![
            response(404, Vec::new()),
            response(200, serde_json::to_vec(&created).unwrap()),
            response(200, pod_json(0.75)),
            response(204, Vec::new()),
        ]);
        let client = client(&base, 8192);
        let scheduled = ScheduledPod {
            schema_version: "livefire.rag.runpod-scheduled-pod/1",
            pod_id: "pod-1".into(),
            terminate_after: "2026-08-18T00:00:00Z".into(),
            termination_binding: "requested_unobservable",
            graphql_request_sha256: rag_pipeline::digest_bytes(b"request"),
            graphql_request: json!({"test":"request"}),
        };
        let admitted = client
            .admit_scheduled_pod(&scheduled, &pod_spec())
            .await
            .unwrap();
        assert_eq!(admitted.desired_status, PodDesiredStatus::Running);
        client.delete_pod(&admitted.id).await.unwrap();
        for _ in 0..3 {
            assert!(requests.recv().unwrap().starts_with("GET /v1/pods/pod-1"));
        }
        assert!(
            requests
                .recv()
                .unwrap()
                .starts_with("DELETE /v1/pods/pod-1")
        );
        server.join().unwrap();
    }

    #[test]
    fn graphql_worker_arguments_reject_every_shell_metacharacter() {
        for metacharacter in [
            ';', '$', '&', '|', '(', ')', '<', '>', '*', '?', '!', '`', '\\',
        ] {
            let mut specification = pod_spec();
            specification.worker_arguments = vec![format!("run{metacharacter}unsafe")];
            assert!(
                build_graphql_pod_request(&specification, "2026-08-18T00:00:00Z").is_err(),
                "accepted GraphQL dockerArgs metacharacter {metacharacter:?}"
            );
        }
    }

    #[tokio::test]
    async fn scheduled_pod_identity_mismatch_is_deleted_immediately() {
        let mut wrong_gpu: Value = serde_json::from_slice(&pod_json(0.75)).unwrap();
        wrong_gpu["gpu"]["id"] = Value::String("NVIDIA A40".into());
        wrong_gpu["gpu"]["displayName"] = Value::String("NVIDIA A40".into());
        let (base, requests, server) = spawn_server(vec![
            response(200, serde_json::to_vec(&wrong_gpu).unwrap()),
            response(204, Vec::new()),
        ]);
        let client = client(&base, 8192);
        let scheduled = ScheduledPod {
            schema_version: "livefire.rag.runpod-scheduled-pod/1",
            pod_id: "pod-1".into(),
            terminate_after: "2026-08-18T00:00:00Z".into(),
            termination_binding: "requested_unobservable",
            graphql_request_sha256: rag_pipeline::digest_bytes(b"request"),
            graphql_request: json!({"test":"request"}),
        };
        assert!(
            client
                .admit_scheduled_pod(&scheduled, &pod_spec())
                .await
                .is_err()
        );
        assert!(requests.recv().unwrap().starts_with("GET /v1/pods/pod-1"));
        assert!(
            requests
                .recv()
                .unwrap()
                .starts_with("DELETE /v1/pods/pod-1")
        );
        server.join().unwrap();
    }

    #[tokio::test]
    async fn price_cap_and_identity_mismatch_delete_the_created_pod() {
        for mutation in ["price", "volume", "gpu", "cloud", "image"] {
            let mut pod = serde_json::from_slice::<Value>(&pod_json(if mutation == "price" {
                1.01
            } else {
                0.5
            }))
            .unwrap();
            match mutation {
                "volume" => pod["networkVolume"]["id"] = json!("wrong-volume"),
                "gpu" => pod["gpu"]["id"] = json!("NVIDIA L40S"),
                "cloud" => pod["machine"]["secureCloud"] = json!(false),
                "image" => {
                    pod["image"] = json!(
                        "ghcr.io/example/other@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                    )
                }
                "price" => {}
                _ => unreachable!(),
            }
            let (base, requests, server) = spawn_server(vec![
                response(201, serde_json::to_vec(&pod).unwrap()),
                response(204, Vec::new()),
            ]);
            let error = client(&base, 8192)
                .create_pod(&pod_spec())
                .await
                .unwrap_err();
            assert!(matches!(
                error,
                RunpodControlError::CreatedPodRejected {
                    cleanup: CleanupOutcome::Succeeded,
                    ..
                }
            ));
            assert!(format!("{error}").contains("cleanup succeeded"));
            assert!(requests.recv().unwrap().starts_with("POST /v1/pods "));
            assert!(
                requests
                    .recv()
                    .unwrap()
                    .starts_with("DELETE /v1/pods/pod-1 ")
            );
            server.join().unwrap();
        }
    }

    #[tokio::test]
    async fn cleanup_failure_is_reported_without_hiding_the_original_rejection() {
        let (base, _, server) = spawn_server(vec![
            response(201, pod_json(2.0)),
            response(500, b"no".to_vec()),
        ]);
        let error = client(&base, 8192)
            .create_pod(&pod_spec())
            .await
            .unwrap_err();
        assert!(
            matches!(
                error,
                RunpodControlError::CreatedPodRejected {
                    cleanup: CleanupOutcome::Failed,
                    ..
                }
            ),
            "{error:?}"
        );
        assert!(format!("{error}").contains("cleanup failed"));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn non_success_malformed_and_oversized_responses_fail_closed() {
        for (fake, maximum, expected) in [
            (response(400, b"bad".to_vec()), 1024, "status"),
            (response(201, b"not-json".to_vec()), 1024, "JSON"),
            (response(201, vec![b'x'; 129]), 128, "exceeded"),
            (
                FakeResponse {
                    status: 201,
                    body: Vec::new(),
                    declared_length: Some(129),
                },
                128,
                "exceeded",
            ),
            (
                response(
                    201,
                    serde_json::to_vec(&{
                        let mut pod = serde_json::from_slice::<Value>(&pod_json(0.5)).unwrap();
                        pod["desiredStatus"] = json!("UNKNOWN");
                        pod
                    })
                    .unwrap(),
                ),
                4096,
                "JSON",
            ),
        ] {
            let (base, _, server) = spawn_server(vec![fake]);
            let error = client(&base, maximum)
                .create_pod(&pod_spec())
                .await
                .unwrap_err();
            assert!(format!("{error}").contains(expected));
            server.join().unwrap();
        }
    }

    #[test]
    fn dry_run_and_debug_are_secret_free_and_image_must_be_pinned() {
        let client = client("http://127.0.0.1:1/v1", 4096);
        let dry_run = client.dry_run_create_pod(&pod_spec()).unwrap();
        let rendered = serde_json::to_string(&dry_run).unwrap();
        let debug = format!("{client:?}");
        assert!(!rendered.contains(SECRET));
        assert!(!debug.contains(SECRET));
        assert!(rendered.contains("<redacted>"));
        assert_eq!(dry_run.pointer("/body/env"), Some(&json!({})));

        let mut unpinned = pod_spec();
        unpinned.image = "ghcr.io/huggingface/text-embeddings-inference:latest".into();
        assert!(client.dry_run_create_pod(&unpinned).is_err());
    }

    #[test]
    fn named_environment_variable_is_required_without_echoing_its_value() {
        let error = RunpodClient::from_environment(
            "LIVEFIRE_RUNPOD_TEST_KEY_THAT_IS_NOT_SET",
            RunpodClientLimits::default(),
        )
        .unwrap_err();
        assert!(matches!(error, RunpodControlError::MissingApiKey { .. }));
        assert!(!format!("{error:?} {error}").contains(SECRET));
    }

    #[test]
    fn error_messages_never_echo_api_key_material() {
        for error in [
            RunpodControlError::Transport { operation: "test" },
            RunpodControlError::UnexpectedStatus {
                operation: "test",
                status: 401,
            },
            RunpodControlError::MalformedResponse { operation: "test" },
            RunpodControlError::MalformedJsonResponse {
                operation: "test",
                reason: "missing field".into(),
            },
            RunpodControlError::CreatedPodRejected {
                reason: "test",
                cleanup: CleanupOutcome::Failed,
            },
        ] {
            let rendered = format!("{error:?} {error}");
            assert!(!rendered.contains(SECRET));
            assert!(!rendered.contains("Bearer runpod"));
        }
    }
}
