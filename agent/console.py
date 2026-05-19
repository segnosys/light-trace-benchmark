"""
Console / terminal helpers for the live agent driver.

Originally embedded inline in `agent_throughput.py`. Two things live here:

  - `Colors`: a tiny ANSI escape-code namespace used for the live dashboard line.
  - `KeyTap`: a background thread that listens for keypresses while the
    benchmark is running so the user can interactively trigger a fresh
    session (pressing 'n').

`KeyTap` is platform-dependent — it uses termios + tty + select, all of
which are POSIX-only. On Windows it falls back to a noop quietly.
"""
import select
import sys
import threading


class Colors:
    """ANSI escape codes used throughout the live dashboard line."""

    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


class KeyTap:
    """Background thread that listens for keypresses during benchmark runs.

    Pressing `n` queues one additional "force new session" request. The
    main loop polls `get_pending_count()` between turns and honors any
    queued requests by spawning fresh sessions instead of growing an
    existing one. This makes it easy to demo cache-hit/miss behavior
    live without restarting the run.

    Platform note: termios+tty are POSIX-only. On non-POSIX hosts the
    listener silently does nothing instead of raising — the benchmark
    still works, just without interactive keyboard control.
    """

    def __init__(self):
        self.pending_new_sessions = 0
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.old_settings = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()
        print(f"{Colors.YELLOW}Press n to force next request to create a new session{Colors.END}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)

    def _listen(self):
        """Background thread that listens for keypresses."""
        try:
            import termios
            import tty
        except ImportError:
            # Non-POSIX (Windows); keypress feature is a no-op.
            return
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

            while self.running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    if char.lower() == "n":
                        with self.lock:
                            self.pending_new_sessions += 1
                        print(
                            f"\n{Colors.YELLOW}>>> Queued additional new session request{Colors.END}"
                        )
        except Exception:
            # Not running in a real terminal (e.g. CI capture, pytest).
            pass
        finally:
            if self.old_settings:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                except (termios.error, OSError, AttributeError):
                    pass

    def get_pending_count(self) -> int:
        """Atomically return queued new-session count and reset to 0."""
        with self.lock:
            count = self.pending_new_sessions
            self.pending_new_sessions = 0
            return count
