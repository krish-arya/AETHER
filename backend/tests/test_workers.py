from app.core.config import APP_VERSION
from app.workers.celery_app import celery_app, ping


def test_ping_task_is_registered_and_runs_eagerly():
    assert "aether.system.ping" in celery_app.tasks
    result = ping.apply().get()
    assert result["status"] == "ok"
    assert result["version"] == APP_VERSION


def test_celery_uses_json_serialisation_only():
    assert celery_app.conf.task_serializer == "json"
    assert celery_app.conf.accept_content == ["json"]
