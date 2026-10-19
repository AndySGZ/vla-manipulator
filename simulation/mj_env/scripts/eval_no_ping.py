"""Run the existing evaluator without WebSocket keepalive during JAX compilation.

This is the helper used by the state-input 25k / max600 server evaluation.
It changes the client's connection keepalive, not the policy or action loop.
"""

import functools
import os
import runpy
import sys

import websockets.sync.client

sys.path.insert(0, os.getcwd())
websockets.sync.client.connect = functools.partial(
    websockets.sync.client.connect, ping_interval=None
)
runpy.run_module("mj_env.scripts.evaluate_openpi", run_name="__main__")
