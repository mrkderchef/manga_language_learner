"""Central logging configuration for the backend."""

from __future__ import annotations

import logging
import sys
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """Install one predictable logging setup, even under uvicorn reload."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    for name in ("uvicorn", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True

    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers.clear()
    access_logger.propagate = False
    access_logger.disabled = True

    for name in ("httpx", "huggingface_hub", "PIL", "multipart"):
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True


def _is_important_success(method: str, path: str) -> bool:
    if method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return False
    if path.startswith(("/api/media/", "/api/thumb/", "/css/", "/js/", "/assets/")):
        return False
    return path.startswith(("/api/scanner/", "/api/runtime/ocr-assets/"))


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log important requests with the same formatter as application events."""

    async def dispatch(self, request, call_next):
        request_id = uuid.uuid4().hex[:8]
        request.state.request_id = request_id
        method = request.method.upper()
        path = request.url.path
        started = time.perf_counter()
        logger = logging.getLogger("request")

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            logger.exception(
                'request_id=%s method=%s path=%s status=error duration_ms=%s msg="HTTP request failed"',
                request_id,
                method,
                path,
                duration_ms,
            )
            raise

        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        response.headers["X-Request-ID"] = request_id
        should_log = response.status_code >= 400 or _is_important_success(method, path)
        if should_log:
            level = logging.WARNING if response.status_code >= 400 else logging.INFO
            logger.log(
                level,
                'request_id=%s method=%s path=%s status=%s duration_ms=%s msg="HTTP request done"',
                request_id,
                method,
                path,
                response.status_code,
                duration_ms,
            )
        return response
