# -*- coding: utf-8 -*-
"""POST /secure-request 的 nonce 防重放表容量行为测试。

覆盖：
- 容量上限：占位与已成功记录共用一张有容量上限的表，表满且无过期项时
  新 nonce 返回 503 nonce_capacity_exceeded，不驱逐未过期 nonce、
  不执行 payload、不消耗新 nonce 的名额；
- 重放优先：表满时重放已成功的 nonce 仍返回 409（优先于 503）；
- 占位回滚：payload 字段校验失败（422 invalid_request）与业务语义失败
  （422 unsatisfiable）都释放自身占位，容量不泄漏，同一 nonce 可重试；
- 过期清理：过期记录被清理后容量即刻恢复；
- 安全释放：_release_nonce_placeholder 只删除本请求写入的占位，不误删
  占位过期后被并发请求重新占用的记录；
- 原有语义保留：验签 / 时间窗 / 未知 key_id / 未配置密钥的行为不变；
- 配置加载：SECURE_REQUEST_NONCE_CAPACITY 默认 10000，必须为正整数。

直接运行：python -m unittest discover -s tests
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
import unittest
from unittest import mock

from app.main import (
    SecureRequestConfig,
    _nonce_lock,
    _nonce_seen,
    _release_nonce_placeholder,
    app,
    canonical_json,
    secure_config,
)

try:
    from starlette.testclient import TestClient
except ImportError:  # pragma: no cover
    TestClient = None  # type: ignore[assignment]

KEY_ID = "demo-key"
SECRET = b"0123456789abcdef0123456789abcdef"

PLAN_PAYLOAD = {
    "object_length": 256,
    "ranges": [{"type": "interval", "start": 0, "end": 99}],
    "max_gap": 8,
    "fetch_limit": 128,
}

# 缺少 fetch_limit：模型校验失败（422 invalid_request）。
BAD_SCHEMA_PAYLOAD = {
    "object_length": 256,
    "ranges": [{"type": "interval", "start": 0, "end": 99}],
    "max_gap": 8,
}

# 区间越界：业务语义失败（422 unsatisfiable，走异常回滚路径）。
UNSATISFIABLE_PAYLOAD = {
    "object_length": 10,
    "ranges": [{"type": "interval", "start": 0, "end": 99}],
    "max_gap": 8,
    "fetch_limit": 128,
}


def make_envelope(
    payload: dict,
    nonce: str,
    *,
    key_id: str = KEY_ID,
    timestamp: int | None = None,
    secret: bytes = SECRET,
) -> dict:
    """按服务端相同的规范化构造签名信封。"""
    envelope = {
        "operation": "plan",
        "payload": payload,
        "key_id": key_id,
        "timestamp": int(time.time()) if timestamp is None else timestamp,
        "nonce": nonce,
    }
    envelope["signature"] = hmac.new(
        secret, canonical_json(envelope), hashlib.sha256
    ).hexdigest()
    return envelope


@unittest.skipIf(TestClient is None, "需要安装 starlette（随 fastapi 安装）")
class SecureRequestNonceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self._saved = (
            dict(secure_config.keys),
            secure_config.timestamp_skew,
            secure_config.nonce_ttl,
            secure_config.nonce_capacity,
        )
        secure_config.keys = {KEY_ID: SECRET}
        secure_config.timestamp_skew = 300
        secure_config.nonce_ttl = 600
        secure_config.nonce_capacity = 10000
        with _nonce_lock:
            _nonce_seen.clear()

    def tearDown(self) -> None:
        keys, skew, ttl, capacity = self._saved
        secure_config.keys = keys
        secure_config.timestamp_skew = skew
        secure_config.nonce_ttl = ttl
        secure_config.nonce_capacity = capacity
        with _nonce_lock:
            _nonce_seen.clear()

    def post(self, envelope: dict):
        return self.client.post("/secure-request", json=envelope)

    # ------------------------- 原有语义保留 -------------------------

    def test_success_returns_plan_and_occupies_nonce(self) -> None:
        resp = self.post(make_envelope(PLAN_PAYLOAD, "nonce-1"))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["object_length"], 256)
        self.assertIn((KEY_ID, "nonce-1"), _nonce_seen)

    def test_replay_successful_nonce_409(self) -> None:
        envelope = make_envelope(PLAN_PAYLOAD, "nonce-1")
        self.assertEqual(self.post(envelope).status_code, 200)
        resp = self.post(envelope)
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json()["error"], "nonce_replayed")

    def test_invalid_signature_401(self) -> None:
        envelope = make_envelope(PLAN_PAYLOAD, "nonce-1")
        envelope["signature"] = "0" * 64
        resp = self.post(envelope)
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json()["error"], "invalid_signature")
        self.assertEqual(len(_nonce_seen), 0)

    def test_timestamp_skew_401(self) -> None:
        envelope = make_envelope(
            PLAN_PAYLOAD, "nonce-1", timestamp=int(time.time()) - 10000
        )
        resp = self.post(envelope)
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json()["error"], "timestamp_skew")

    def test_unknown_key_id_401(self) -> None:
        resp = self.post(make_envelope(PLAN_PAYLOAD, "nonce-1", key_id="ghost"))
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json()["error"], "unknown_key_id")

    def test_keys_not_configured_503(self) -> None:
        secure_config.keys = {}
        resp = self.post(make_envelope(PLAN_PAYLOAD, "nonce-1"))
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.json()["error"], "keys_not_configured")

    # ------------------------- 容量上限 -------------------------

    def test_capacity_exceeded_503_without_eviction_or_execution(self) -> None:
        secure_config.nonce_capacity = 2
        self.assertEqual(self.post(make_envelope(PLAN_PAYLOAD, "n1")).status_code, 200)
        self.assertEqual(self.post(make_envelope(PLAN_PAYLOAD, "n2")).status_code, 200)

        resp = self.post(make_envelope(PLAN_PAYLOAD, "n3"))
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.json()["error"], "nonce_capacity_exceeded")
        self.assertEqual(resp.json()["nonce_capacity"], 2)

        # 不驱逐未过期 nonce、不为被拒绝的 nonce 消耗名额。
        self.assertEqual(len(_nonce_seen), 2)
        self.assertNotIn((KEY_ID, "n3"), _nonce_seen)
        replay = self.post(make_envelope(PLAN_PAYLOAD, "n1"))
        self.assertEqual(replay.status_code, 409)
        self.assertEqual(replay.json()["error"], "nonce_replayed")

    def test_replay_takes_priority_over_capacity(self) -> None:
        secure_config.nonce_capacity = 1
        self.assertEqual(self.post(make_envelope(PLAN_PAYLOAD, "n1")).status_code, 200)
        # 表已满，但重放已成功的 nonce 仍报 409 而非 503。
        resp = self.post(make_envelope(PLAN_PAYLOAD, "n1"))
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json()["error"], "nonce_replayed")

    def test_expired_entries_are_swept_freeing_capacity(self) -> None:
        secure_config.nonce_capacity = 1
        self.assertEqual(self.post(make_envelope(PLAN_PAYLOAD, "n1")).status_code, 200)
        # 模拟保留窗口过去：记录已过期，应被清理并释放容量。
        with _nonce_lock:
            _nonce_seen[(KEY_ID, "n1")] = time.time() - 1
        resp = self.post(make_envelope(PLAN_PAYLOAD, "n2"))
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn((KEY_ID, "n1"), _nonce_seen)
        self.assertIn((KEY_ID, "n2"), _nonce_seen)

    # ------------------------- 占位回滚 -------------------------

    def test_schema_error_releases_placeholder_and_capacity(self) -> None:
        secure_config.nonce_capacity = 1
        resp = self.post(make_envelope(BAD_SCHEMA_PAYLOAD, "n1"))
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"], "invalid_request")
        self.assertNotIn((KEY_ID, "n1"), _nonce_seen)
        # 同一 nonce 修正后可重试成功（容量未被失败请求泄漏）。
        self.assertEqual(self.post(make_envelope(PLAN_PAYLOAD, "n1")).status_code, 200)

    def test_business_error_releases_placeholder_and_capacity(self) -> None:
        secure_config.nonce_capacity = 1
        resp = self.post(make_envelope(UNSATISFIABLE_PAYLOAD, "n1"))
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"], "unsatisfiable")
        self.assertNotIn((KEY_ID, "n1"), _nonce_seen)
        # 容量未泄漏：全新 nonce 仍可成功。
        self.assertEqual(self.post(make_envelope(PLAN_PAYLOAD, "n2")).status_code, 200)

    # ------------------------- 安全释放 -------------------------

    def test_release_placeholder_only_removes_own_entry(self) -> None:
        key = (KEY_ID, "nonce-x")
        with _nonce_lock:
            # 本请求占位过期后，同一 nonce 被并发请求以更新的到期时间重新占用。
            _nonce_seen[key] = 2000.0
        _release_nonce_placeholder(key, 1000.0)  # 本请求的占位值
        self.assertEqual(_nonce_seen.get(key), 2000.0)  # 不得误删他人记录

        _release_nonce_placeholder(key, 2000.0)  # 值匹配才删除
        self.assertNotIn(key, _nonce_seen)

        _release_nonce_placeholder(key, 2000.0)  # 记录不存在时不报错


class SecureRequestConfigTests(unittest.TestCase):
    """SECURE_REQUEST_NONCE_CAPACITY 的加载与校验。"""

    def test_default_capacity_is_10000(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("SECURE_REQUEST_NONCE_CAPACITY", None)
            self.assertEqual(SecureRequestConfig().nonce_capacity, 10000)

    def test_capacity_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"SECURE_REQUEST_NONCE_CAPACITY": "42"}):
            self.assertEqual(SecureRequestConfig().nonce_capacity, 42)

    def test_capacity_must_be_positive_integer(self) -> None:
        for bad in ("0", "-3"):
            with mock.patch.dict(
                os.environ, {"SECURE_REQUEST_NONCE_CAPACITY": bad}
            ):
                with self.assertRaises(RuntimeError):
                    SecureRequestConfig()


if __name__ == "__main__":
    unittest.main()
