from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import make_asgi_app
from starlette.routing import Route

from litellm.integrations.prometheus import PrometheusLogger


def _register_metrics_routes(app: FastAPI) -> None:
    metrics_app = make_asgi_app()
    handler = PrometheusLogger._prometheus_metrics_route_handler(metrics_app)
    app.routes.append(Route("/metrics", endpoint=handler, methods=["GET", "HEAD"]))
    app.routes.append(Route("/metrics/", endpoint=handler, methods=["GET", "HEAD"]))


def test_metrics_endpoint_serves_without_trailing_slash_redirect():
    app = FastAPI()
    _register_metrics_routes(app)
    client = TestClient(app)

    response = client.get("/metrics", follow_redirects=False)

    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")


def test_metrics_endpoint_with_trailing_slash_still_works():
    app = FastAPI()
    _register_metrics_routes(app)
    client = TestClient(app)

    response = client.get("/metrics/", follow_redirects=False)

    assert response.status_code == 200


def test_mount_metrics_endpoint_is_idempotent():
    from unittest.mock import patch

    test_app = FastAPI()
    initial = len(test_app.routes)
    with patch("litellm.proxy.proxy_server.app", test_app):
        PrometheusLogger._mount_metrics_endpoint()
        after_first = len(test_app.routes)
        PrometheusLogger._mount_metrics_endpoint()
        after_second = len(test_app.routes)

    assert after_first == initial + 2
    assert after_second == after_first
