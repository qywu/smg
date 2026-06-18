"""Tests for the shared Prometheus ``/metrics`` sidecar (engine-free).

``smg_grpc_servicer.metrics`` depends only on stdlib + prometheus_client, so
these run without any inference engine installed. They drive the async sidecar
via ``asyncio.run`` so no pytest-asyncio config is required.

Run with: pytest grpc_servicer/tests/test_metrics_sidecar.py
"""

from __future__ import annotations

import asyncio
import urllib.error
import urllib.request

import pytest

pytest.importorskip("prometheus_client")

from prometheus_client import CollectorRegistry  # noqa: E402
from prometheus_client.parser import text_string_to_metric_families  # noqa: E402
from smg_grpc_servicer.metrics import (  # noqa: E402
    METRICS_PORT_ENV,
    SchedulerLoadCollector,
    metrics_server_args,
    metrics_url,
    resolve_metrics_port,
    start_metrics_sidecar,
)


def _get(url: str) -> tuple[int, str, str]:
    with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310 — loopback only.
        return resp.status, resp.headers.get("Content-Type", ""), resp.read().decode()


async def _fetch(loop, url: str) -> tuple[int, str, str]:
    """Run the blocking urlopen off the event loop so the sidecar can serve it."""
    return await loop.run_in_executor(None, _get, url)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_metrics_url_skips_wildcard_hosts():
    assert metrics_url("0.0.0.0", 9100) is None
    assert metrics_url("::", 9100) is None
    assert metrics_url("", 9100) is None


def test_metrics_url_for_routable_host():
    assert metrics_url("10.1.2.3", 9100) == "http://10.1.2.3:9100/metrics"


def test_metrics_server_args_disabled_is_empty():
    assert metrics_server_args("10.1.2.3", None) == {}


def test_metrics_server_args_routable_host_advertises_port_and_url():
    args = metrics_server_args("10.1.2.3", 9100)
    assert args == {"metrics_port": 9100, "metrics_url": "http://10.1.2.3:9100/metrics"}


def test_metrics_server_args_wildcard_host_advertises_port_only():
    # Gateway combines metrics_port with the worker address it discovered.
    assert metrics_server_args("0.0.0.0", 9100) == {"metrics_port": 9100}


def test_resolve_metrics_port_prefers_explicit():
    assert resolve_metrics_port(9100) == 9100


def test_resolve_metrics_port_none_without_env(monkeypatch):
    monkeypatch.delenv(METRICS_PORT_ENV, raising=False)
    assert resolve_metrics_port(None) is None


def test_resolve_metrics_port_reads_env(monkeypatch):
    monkeypatch.setenv(METRICS_PORT_ENV, "9200")
    assert resolve_metrics_port(None) == 9200


@pytest.mark.parametrize("value", ["abc", "0", "70000", "-1"])
def test_resolve_metrics_port_rejects_bad_env(monkeypatch, value):
    monkeypatch.setenv(METRICS_PORT_ENV, value)
    assert resolve_metrics_port(None) is None


# ---------------------------------------------------------------------------
# SchedulerLoadCollector
# ---------------------------------------------------------------------------


def test_scheduler_load_collector_emits_gauges():
    snapshot = {
        "num_running_reqs": 3,
        "num_waiting_reqs": 2,
        "num_total_reqs": 5,
        "token_usage": 0.42,
    }
    registry = CollectorRegistry()
    registry.register(SchedulerLoadCollector(lambda: snapshot))

    assert registry.get_sample_value("smg_scheduler_running_requests") == 3.0
    assert registry.get_sample_value("smg_scheduler_waiting_requests") == 2.0
    assert registry.get_sample_value("smg_scheduler_total_requests") == 5.0
    assert registry.get_sample_value("smg_scheduler_token_usage") == 0.42


def test_scheduler_load_collector_tolerates_missing_keys():
    registry = CollectorRegistry()
    registry.register(SchedulerLoadCollector(dict))  # empty snapshot
    assert registry.get_sample_value("smg_scheduler_running_requests") == 0.0
    assert registry.get_sample_value("smg_scheduler_token_usage") == 0.0


# ---------------------------------------------------------------------------
# HTTP sidecar
# ---------------------------------------------------------------------------


def test_metrics_endpoint_returns_valid_exposition():
    async def scenario():
        registry = CollectorRegistry()
        registry.register(
            SchedulerLoadCollector(
                lambda: {
                    "num_running_reqs": 7,
                    "num_waiting_reqs": 1,
                    "num_total_reqs": 8,
                    "token_usage": 0.5,
                }
            )
        )
        sidecar = await start_metrics_sidecar("127.0.0.1", 0, registry=registry)
        assert sidecar is not None
        loop = asyncio.get_running_loop()
        try:
            status, content_type, body = await _fetch(
                loop, f"http://127.0.0.1:{sidecar.port}/metrics"
            )
        finally:
            await sidecar.close()
        return status, content_type, body

    status, content_type, body = asyncio.run(scenario())
    assert status == 200
    assert content_type.startswith("text/plain")
    families = {f.name: f for f in text_string_to_metric_families(body)}
    assert "smg_scheduler_running_requests" in families
    assert families["smg_scheduler_running_requests"].samples[0].value == 7.0


def test_root_path_also_serves_metrics():
    async def scenario():
        sidecar = await start_metrics_sidecar("127.0.0.1", 0, registry=CollectorRegistry())
        assert sidecar is not None
        loop = asyncio.get_running_loop()
        try:
            return await _fetch(loop, f"http://127.0.0.1:{sidecar.port}/")
        finally:
            await sidecar.close()

    status, _, body = asyncio.run(scenario())
    assert status == 200
    # Empty registry still yields valid (possibly empty) exposition.
    list(text_string_to_metric_families(body))


def test_unknown_path_returns_404():
    async def scenario():
        sidecar = await start_metrics_sidecar("127.0.0.1", 0, registry=CollectorRegistry())
        assert sidecar is not None
        loop = asyncio.get_running_loop()
        try:
            await _fetch(loop, f"http://127.0.0.1:{sidecar.port}/healthz")
        finally:
            await sidecar.close()

    with pytest.raises(urllib.error.HTTPError) as exc:
        asyncio.run(scenario())
    assert exc.value.code == 404


def test_query_string_is_ignored():
    async def scenario():
        sidecar = await start_metrics_sidecar("127.0.0.1", 0, registry=CollectorRegistry())
        assert sidecar is not None
        loop = asyncio.get_running_loop()
        try:
            return await _fetch(loop, f"http://127.0.0.1:{sidecar.port}/metrics?foo=bar")
        finally:
            await sidecar.close()

    status, _, _ = asyncio.run(scenario())
    assert status == 200


def test_start_is_best_effort_on_bind_failure():
    async def scenario():
        registry = CollectorRegistry()
        first = await start_metrics_sidecar("127.0.0.1", 0, registry=registry)
        assert first is not None
        try:
            # Second bind to the same port must fail-soft (return None).
            second = await start_metrics_sidecar("127.0.0.1", first.port, registry=registry)
            return second
        finally:
            await first.close()

    assert asyncio.run(scenario()) is None


def test_close_is_idempotent():
    async def scenario():
        sidecar = await start_metrics_sidecar("127.0.0.1", 0, registry=CollectorRegistry())
        assert sidecar is not None
        await sidecar.close()
        await sidecar.close()  # second close must not raise

    asyncio.run(scenario())
