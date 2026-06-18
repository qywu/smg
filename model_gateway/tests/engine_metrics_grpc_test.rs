//! Integration tests for `GET /engine_metrics` covering gRPC workers (W1).
//!
//! A gRPC worker has no HTTP base URL, so its scrape endpoint is carried in the
//! `metrics_url` label (populated by metadata discovery). These tests stand up a
//! stub HTTP `/metrics` server, register a gRPC-mode worker pointing at it, and
//! assert the worker's series merge into the aggregated `/engine_metrics` output.

use std::{collections::HashMap, sync::Arc, time::Duration};

use axum::{response::IntoResponse, routing::get, Router};
use smg::worker::{
    manager::EngineMetricsResult, BasicWorkerBuilder, ConnectionMode, Worker, WorkerManager,
    WorkerRegistry, WorkerType,
};
use tokio::sync::oneshot;

/// Stub HTTP server exposing a fixed `/metrics` body. Returns its base URL and a
/// shutdown handle.
struct StubMetricsServer {
    url: String,
    shutdown: Option<oneshot::Sender<()>>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl StubMetricsServer {
    #[expect(
        clippy::expect_used,
        clippy::disallowed_methods,
        reason = "test infrastructure - panicking on setup failure is intentional"
    )]
    async fn start(body: &'static str) -> Self {
        // Bind once and serve the same listener: reading the port and rebinding
        // would race another process onto it between the two binds.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind stub metrics server");
        let port = listener.local_addr().expect("local addr").port();

        let app = Router::new().route("/metrics", get(move || async move { body.into_response() }));
        let (tx, rx) = oneshot::channel::<()>();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = rx.await;
                })
                .await
                .expect("stub server");
        });

        Self {
            url: format!("http://127.0.0.1:{port}"),
            shutdown: Some(tx),
            handle: Some(handle),
        }
    }

    async fn stop(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
        }
    }
}

fn grpc_worker(url: &str, labels: &[(&str, &str)]) -> Arc<dyn Worker> {
    let labels: HashMap<String, String> = labels
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .labels(labels)
            .build(),
    )
}

const GRPC_METRICS: &str = "\
# HELP sglang_num_running_reqs The number of running requests
# TYPE sglang_num_running_reqs gauge
sglang_num_running_reqs{model_name=\"m\"} 7
";

#[tokio::test]
async fn grpc_worker_metrics_merge_into_engine_metrics() {
    let stub = StubMetricsServer::start(GRPC_METRICS).await;
    let metrics_url = format!("{}/metrics", stub.url);

    let registry = WorkerRegistry::new();
    registry
        .register(grpc_worker(
            "grpc://127.0.0.1:30001",
            &[("metrics_url", metrics_url.as_str())],
        ))
        .expect("register grpc worker");

    let client = reqwest::Client::new();
    let result = WorkerManager::get_engine_metrics(&registry, &client).await;

    let text = match result {
        EngineMetricsResult::Ok(text) => text,
        EngineMetricsResult::Err(msg) => panic!("expected Ok, got Err: {msg}"),
    };

    // The gRPC worker's series merged in, tagged with its gRPC identity.
    assert!(
        text.contains("sglang_num_running_reqs"),
        "missing gRPC series:\n{text}"
    );
    assert!(
        text.contains("grpc://127.0.0.1:30001"),
        "missing worker_addr label:\n{text}"
    );

    stub.stop().await;
}

#[tokio::test]
async fn grpc_workers_without_endpoint_are_skipped_not_failed() {
    // An all-gRPC fleet with no discovered metrics endpoint must report the
    // real cause ("no endpoint"), not the misleading "all backend requests
    // failed" that the HTTP-only fetch used to produce.
    let registry = WorkerRegistry::new();
    registry
        .register(grpc_worker("grpc://127.0.0.1:30001", &[]))
        .expect("register grpc worker");

    let client = reqwest::Client::new();
    let result = WorkerManager::get_engine_metrics(&registry, &client).await;

    match result {
        EngineMetricsResult::Err(msg) => {
            assert!(
                msg.contains("metrics endpoint"),
                "unexpected error message: {msg}"
            );
        }
        EngineMetricsResult::Ok(text) => panic!("expected Err, got Ok:\n{text}"),
    }
}
