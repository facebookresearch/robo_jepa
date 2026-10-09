# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
gym_tcp.py — Lightweight ZMQ-based server/client for Gym-compatible robot environments.

Drop-in replacement for agentlace.gym_env.GymEnvServerWrapper / GymEnvClient.
Uses ZMQ REQ/REP for reliable synchronous request-reply.

Serialisation
-------------
Messages are encoded with **MessagePack** (msgpack), which is a binary format
with implementations in every major language (Python, C++, Rust, Go, JS, …).
Numpy arrays are encoded as a tagged map so non-Python clients can decode them:

    {"__ndarray__": true, "dtype": "float32", "shape": [7], "data": <bytes>}

All other types follow standard msgpack encoding (ints, floats, strings, bytes,
lists, maps).

Protocol
--------
Every message is a msgpack-encoded dict sent over a ZMQ REQ/REP socket pair.

Client → Server request:
    {"type": "reset" | "step" | "step_chunk" | "close", "payload": <kwargs_dict | action_array | chunk_dict>}

For ``step_chunk``, payload is::

    {"actions": <(N, DoF) ndarray>, "dt": <float, seconds per action>}

The server executes all N actions sequentially at the given rate and returns
the final observation plus a list of all intermediate observations.

Server → Client response:
    {"result": <return_value>}       # on success
    {"error": "<traceback string>"}  # on exception inside the env

Usage
-----
Server (robot PC)::

    from gym_tcp import GymEnvServer
    server = GymEnvServer(env, port=4754)
    server.start()          # blocks; pass threaded=True to run in background

Client (agent / cluster)::

    from gym_tcp import GymEnvClient
    env = GymEnvClient(host="localhost", port=4754)
    obs, _ = env.reset()
    obs, reward, done, info = env.step(action)
    env.close()
"""

import threading
import traceback

import msgpack
import numpy as np
import zmq


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _encode_value(obj):
    """Recursively encode a Python object to a msgpack-safe structure.

    Numpy arrays are represented as:
        {"__ndarray__": True, "dtype": "<dtype>", "shape": [d0, d1, ...], "data": bytes}
    """
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": obj.tobytes(),
        }
    if isinstance(obj, np.generic):
        # scalar numpy types (np.float32, np.int64, …) → plain Python scalar
        return obj.item()
    if isinstance(obj, dict):
        return {k: _encode_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode_value(v) for v in obj]
    return obj


def _decode_value(obj):
    """Recursively decode a msgpack-decoded structure back to Python / numpy."""
    if isinstance(obj, dict):
        if obj.get("__ndarray__"):
            return np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"]).copy()
        return {k: _decode_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_value(v) for v in obj]
    return obj


def _pack(obj) -> bytes:
    return msgpack.packb(_encode_value(obj), use_bin_type=True)


def _unpack(data: bytes):
    return _decode_value(msgpack.unpackb(data, raw=False))


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class GymEnvServer:
    """ZMQ REP server that exposes a Gym-compatible environment over the network.

    Handles one request at a time (REP is inherently sequential), which is the
    correct pattern for robot control: the agent waits for the observation before
    sending the next action.

    Args:
        env:       Any object with reset(**kwargs) and step(action) methods.
        port:      TCP port to bind to.
        host:      Interface to bind to. Use "0.0.0.0" to accept remote connections.
    """

    def __init__(self, env, port: int = 4754, host: str = "0.0.0.0"):
        self.env = env
        self.port = port
        self.host = host
        self._stop_event = threading.Event()

    def start(self, threaded: bool = False) -> None:
        """Start the server.

        Args:
            threaded: If False (default), blocks until stop() is called.
                      If True, runs the serve loop in a daemon thread and returns immediately.
        """
        if threaded:
            t = threading.Thread(target=self._serve, daemon=True)
            t.start()
        else:
            self._serve()

    def stop(self) -> None:
        """Signal the serve loop to exit after the current request finishes."""
        self._stop_event.set()

    def _serve(self) -> None:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.bind(f"tcp://{self.host}:{self.port}")
        sock.setsockopt(zmq.RCVTIMEO, 1000)  # 1 s poll interval so stop() is responsive
        print(f"[GymEnvServer] listening on tcp://{self.host}:{self.port}")

        try:
            while not self._stop_event.is_set():
                try:
                    raw = sock.recv()
                except zmq.Again:
                    continue  # timeout — check stop flag and loop

                msg = _unpack(raw)
                msg_type = msg.get("type")
                payload = msg.get("payload")

                try:
                    if msg_type == "reset":
                        result = self.env.reset(**(payload or {}))
                    elif msg_type == "step":
                        result = self.env.step(payload)
                    elif msg_type == "step_chunk":
                        result = self._handle_step_chunk(payload)
                    elif msg_type == "close":
                        result = self.env.close()
                        sock.send(_pack({"result": result}))
                        break
                    else:
                        raise ValueError(f"Unknown message type: {msg_type!r}")
                    sock.send(_pack({"result": result}))
                except Exception:
                    sock.send(_pack({"error": traceback.format_exc()}))
        finally:
            sock.close()
            ctx.term()
            print("[GymEnvServer] stopped.")

    def _handle_step_chunk(self, payload):
        """Execute a chunk of actions at the specified rate.

        Delegates to ``env.step_chunk()`` if available (preferred — the
        wrapper can skip camera reads for intermediate steps). Otherwise
        falls back to calling ``env.step()`` in a loop.
        """
        actions = payload["actions"]
        dt = payload.get("dt", 1.0 / 15)

        if hasattr(self.env, "step_chunk"):
            return self.env.step_chunk(actions, dt=dt)

        # Fallback for envs without step_chunk
        import time as _time

        chunk_obs = []
        result = None
        for action in actions:
            t0 = _time.perf_counter()
            result = self.env.step(action)
            obs = result[0] if isinstance(result, (list, tuple)) else result
            chunk_obs.append(obs)
            elapsed = _time.perf_counter() - t0
            sleep_left = dt - elapsed
            if sleep_left > 0:
                _time.sleep(sleep_left)

        if isinstance(result, (list, tuple)):
            obs, reward, done, info = result
            if not isinstance(info, dict):
                info = {}
            info["chunk_obs"] = chunk_obs
            return (obs, reward, done, info)
        return result


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class GymEnvClient:
    """ZMQ REQ client that talks to a GymEnvServer.

    Provides the same reset() / step() / close() interface as a local Gym env
    so agents require no changes when switching between local and remote envs.

    Args:
        host:         Hostname or IP of the machine running GymEnvServer.
        port:         TCP port the server is bound to.
        timeout_ms:   How long to wait for a server response before raising
                      an error (milliseconds). Default: 60 000 (60 s).
    """

    def __init__(self, host: str = "localhost", port: int = 4754, timeout_ms: int = 60_000):
        self._host = host
        self._port = port
        self._timeout_ms = timeout_ms
        self._ctx = zmq.Context()
        self._sock = self._make_socket()
        print(f"[GymEnvClient] connected to tcp://{host}:{port}")

    def _make_socket(self) -> zmq.Socket:
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        sock.connect(f"tcp://{self._host}:{self._port}")
        return sock

    def _call(self, msg_type: str, payload=None):
        self._sock.send(_pack({"type": msg_type, "payload": payload}))
        raw = self._sock.recv()
        response = _unpack(raw)
        if "error" in response:
            raise RuntimeError(f"[GymEnvClient] server raised an exception:\n{response['error']}")
        return response["result"]

    def reset(self, **kwargs):
        """Reset the environment. kwargs are forwarded to env.reset()."""
        return self._call("reset", kwargs)

    def step(self, action):
        """Send an action and return (obs, reward, done, info)."""
        return self._call("step", action)

    def step_chunk(self, actions, dt=1.0 / 15):
        """Send an action chunk to be executed server-side at the given rate.

        Parameters
        ----------
        actions : list | array-like
            Either a sequence of N actions (any shape) — passed through
            unchanged so per-tick components dicts survive serialisation —
            or a numpy-coercible array of shape ``(N, DoF)``.
        dt : float
            Seconds between actions (default 1/15 ≈ 66 ms).

        Returns
        -------
        obs : dict       — observation after the last action
        reward : None
        done : bool
        info : dict      — info["chunk_obs"] is a list of N obs dicts,
                            one after each action.
        """
        # If `actions` is already a list (e.g. of components dicts), keep it
        # as-is — np.asarray would coerce a list-of-dicts to an OBJECT ndarray
        # which msgpack cannot serialise. Otherwise normalise to ndarray.
        if isinstance(actions, list):
            payload_actions = actions
        else:
            payload_actions = np.asarray(actions)
        payload = {
            "actions": payload_actions,
            "dt": dt,
        }
        return self._call("step_chunk", payload)

    def close(self) -> None:
        """Close the environment and the socket."""
        try:
            self._call("close")
        except Exception:
            pass
        self._sock.close()
        self._ctx.term()
