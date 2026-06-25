"""Wrappers around internal Quote-to-Fulfillment services.

Read-only tools use bounded exponential backoff. Non-idempotent write tools use
a persistent SQLite idempotency ledger and are never retried blindly.
"""

from __future__ import annotations

import functools
import json
import os
import random
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import closing
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, cast

import httpx


P = ParamSpec("P")
R = TypeVar("R")


class ToolError(RuntimeError):
    """Base error raised by a tool wrapper."""


class ToolConfigurationError(ToolError):
    """Raised when a required internal service URL is not configured."""


class TransientToolError(ToolError):
    """Raised for timeouts and temporary upstream failures."""


class AmbiguousSideEffectError(ToolError):
    """Raised when retrying a write could create a duplicate."""


def retry_with_backoff(
    func: Callable[P, R],
    *,
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter_ratio: float = 0.1,
) -> Callable[P, R]:
    """Return a wrapper that retries transient read-only tool failures.

    max_retries counts retries after the first attempt. For example,
    max_retries=3 allows at most four total calls.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if base_delay < 0 or max_delay < 0:
        raise ValueError("retry delays must be non-negative")
    if jitter_ratio < 0:
        raise ValueError("jitter_ratio must be non-negative")

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        delay = base_delay

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except TransientToolError:
                if attempt == max_retries:
                    raise

                bounded_delay = min(delay, max_delay)
                jitter = random.uniform(0.0, bounded_delay * jitter_ratio)
                time.sleep(bounded_delay + jitter)
                delay = min(delay * 2, max_delay)

        raise AssertionError("retry loop exited unexpectedly")

    return wrapper


def _require_url(environment_name: str) -> str:
    """Read a required service URL from the environment."""
    url = os.getenv(environment_name, "").strip()
    if not url:
        raise ToolConfigurationError(
            f"{environment_name} is not configured"
        )
    return url


def _headers(*, idempotency_key: str | None = None) -> dict[str, str]:
    """Build headers shared by internal HTTP services."""
    headers = {"Content-Type": "application/json"}
    token = os.getenv("INTERNAL_API_TOKEN", "").strip()

    if token:
        headers["Authorization"] = f"Bearer {token}"
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key

    return headers


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """POST JSON and normalize transport failures into tool exceptions."""
    timeout_seconds = float(os.getenv("TOOL_TIMEOUT_SECONDS", "10"))

    try:
        response = httpx.post(
            url,
            json=dict(payload),
            headers=_headers(idempotency_key=idempotency_key),
            timeout=timeout_seconds,
        )
    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        raise TransientToolError(f"Request to {url} failed: {exc}") from exc

    if response.status_code in {408, 429} or response.status_code >= 500:
        raise TransientToolError(
            f"Temporary {response.status_code} response from {url}"
        )

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ToolError(
            f"Tool returned HTTP {response.status_code} for {url}"
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise ToolError(f"Tool returned invalid JSON for {url}") from exc

    if not isinstance(data, dict):
        raise ToolError(f"Tool returned a non-object JSON value for {url}")

    return cast(dict[str, Any], data)


def _crm_get_customer(sender_email: str) -> dict[str, Any]:
    return _post_json(
        _require_url("CRM_API_URL"),
        {"sender_email": sender_email},
    )


def _erp_get_stock(sku: str) -> dict[str, Any]:
    return _post_json(
        _require_url("ERP_API_URL"),
        {"sku": sku},
    )


def _calc_price(
    sku: str,
    quantity: int = 1,
    *,
    customer_id: str | None = None,
) -> dict[str, Any]:
    return _post_json(
        _require_url("PRICING_API_URL"),
        {
            "sku": sku,
            "quantity": quantity,
            "customer_id": customer_id,
        },
    )


def _shipping_rate(
    destination: str,
    *,
    weight_kg: float | None = None,
    service_level: str | None = None,
) -> dict[str, Any]:
    return _post_json(
        _require_url("SHIPPING_API_URL"),
        {
            "destination": destination,
            "weight_kg": weight_kg,
            "service_level": service_level,
        },
    )


# These four operations are read-only and therefore safe to retry.
crm_get_customer = retry_with_backoff(_crm_get_customer)
erp_get_stock = retry_with_backoff(_erp_get_stock)
calc_price = retry_with_backoff(_calc_price)
shipping_rate = retry_with_backoff(_shipping_rate)


class IdempotencyLedger:
    """Persistent reservation table for non-idempotent side effects."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._initialization_lock = threading.Lock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.database_path,
            timeout=10,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        if self._initialized:
            return

        with self._initialization_lock:
            if self._initialized:
                return

            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS idempotency_records (
                        idempotency_key TEXT PRIMARY KEY,
                        operation TEXT NOT NULL,
                        status TEXT NOT NULL,
                        result_json TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )

            self._initialized = True

    def reserve(
        self,
        *,
        operation: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        """Reserve a key or return the completed result from an earlier call."""
        self._initialize()
        now = time.time()

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            record = connection.execute(
                """
                SELECT operation, status, result_json
                FROM idempotency_records
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()

            if record is None:
                connection.execute(
                    """
                    INSERT INTO idempotency_records (
                        idempotency_key,
                        operation,
                        status,
                        result_json,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, 'pending', NULL, ?, ?)
                    """,
                    (idempotency_key, operation, now, now),
                )
                connection.commit()
                return None

            if record["operation"] != operation:
                connection.rollback()
                raise ToolError(
                    "Idempotency key was reused for another operation"
                )

            if record["status"] == "completed":
                result_json = record["result_json"] or "{}"
                connection.commit()
                return cast(dict[str, Any], json.loads(result_json))

            connection.rollback()
            raise AmbiguousSideEffectError(
                f"{operation} with this idempotency key is already pending; "
                "manual verification is required before another attempt"
            )

    def complete(
        self,
        *,
        operation: str,
        idempotency_key: str,
        result: Mapping[str, Any],
    ) -> None:
        """Store the successful result for future duplicate calls."""
        self._initialize()
        result_json = json.dumps(dict(result), sort_keys=True)

        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                UPDATE idempotency_records
                SET status = 'completed', result_json = ?, updated_at = ?
                WHERE idempotency_key = ?
                  AND operation = ?
                  AND status = 'pending'
                """,
                (
                    result_json,
                    time.time(),
                    idempotency_key,
                    operation,
                ),
            )

            if cursor.rowcount != 1:
                raise ToolError(
                    "Could not complete idempotency reservation"
                )


_ledger = IdempotencyLedger(
    os.getenv(
        "IDEMPOTENCY_DB_PATH",
        str(Path(__file__).with_name(".data") / "idempotency.sqlite3"),
    )
)


def configure_idempotency_ledger(database_path: str | Path) -> None:
    """Replace the ledger path, primarily for tests and local environments."""
    global _ledger
    _ledger = IdempotencyLedger(database_path)


def _execute_idempotent_write(
    *,
    operation: str,
    idempotency_key: str,
    call: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Execute a write once and cache its result by idempotency key."""
    completed_result = _ledger.reserve(
        operation=operation,
        idempotency_key=idempotency_key,
    )
    if completed_result is not None:
        return completed_result

    try:
        result = call()
    except Exception as exc:
        # The request may have reached the remote service. The reservation stays
        # pending so an automatic retry cannot create a duplicate.
        raise AmbiguousSideEffectError(
            f"{operation} outcome is unknown; manual verification is required"
        ) from exc

    _ledger.complete(
        operation=operation,
        idempotency_key=idempotency_key,
        result=result,
    )
    return result


def create_draft_reply(
    thread_id: str,
    customer_id: str,
    body: str,
    *,
    idempotency_key: str,
) -> dict[str, Any]:
    """Create one draft reply, guarded by a persistent idempotency key."""
    url = _require_url("DRAFT_API_URL")
    return _execute_idempotent_write(
        operation="create_draft_reply",
        idempotency_key=idempotency_key,
        call=lambda: _post_json(
            url,
            {
                "thread_id": thread_id,
                "customer_id": customer_id,
                "body": body,
            },
            idempotency_key=idempotency_key,
        ),
    )


def create_internal_note(
    thread_id: str,
    customer_id: str | None,
    body: str,
    *,
    idempotency_key: str,
) -> dict[str, Any]:
    """Create one internal note, guarded by a persistent idempotency key."""
    url = _require_url("NOTE_API_URL")
    return _execute_idempotent_write(
        operation="create_internal_note",
        idempotency_key=idempotency_key,
        call=lambda: _post_json(
            url,
            {
                "thread_id": thread_id,
                "customer_id": customer_id,
                "body": body,
            },
            idempotency_key=idempotency_key,
        ),
    )
