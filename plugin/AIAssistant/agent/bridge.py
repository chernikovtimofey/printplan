"""Run code on the thread that owns the slicer.

The scene, the settings and slicing may only be touched from the Qt main
thread. The HTTP server and the agent run in worker threads, so every slicer
call goes through `QtMainThreadBridge.call`, which posts the function with
`app.callLater` and waits for its result.
"""
import threading
import traceback


class DirectBridge(object):
    """Calls the function immediately (tests, fake slicer)."""

    def call(self, func, *args, **kwargs):
        return func(*args, **kwargs)


class BridgeTimeout(Exception):
    pass


class QtMainThreadBridge(object):
    def __init__(self, app, timeout=120.0):
        self._app = app
        self._timeout = timeout

    def call(self, func, *args, **kwargs):
        timeout = kwargs.pop("_timeout", self._timeout)
        if threading.current_thread() is threading.main_thread():
            return func(*args, **kwargs)
        box = {}
        done = threading.Event()

        def run():
            try:
                box["result"] = func(*args, **kwargs)
            except BaseException as e:  # re-raised in the calling thread
                box["error"] = e
                box["trace"] = traceback.format_exc()
            finally:
                done.set()

        self._app.callLater(run)
        if not done.wait(timeout):
            raise BridgeTimeout("The slicer did not respond within %.0f s" % timeout)
        if "error" in box:
            raise box["error"]
        return box.get("result")
