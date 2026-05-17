"""
Tests for agent/console.py — Colors + KeyTap (key listener).
"""
import threading
import time


from agent.console import Colors, KeyTap


def test_colors_are_ansi_escape_strings():
    """Sanity: the ANSI codes start with ESC."""
    for name in ("CYAN", "GREEN", "YELLOW", "RED", "BOLD", "DIM", "END"):
        v = getattr(Colors, name)
        assert isinstance(v, str)
        assert v.startswith("\033[")


def test_keytap_starts_and_stops_without_terminal():
    """In a non-tty env (CI, pytest capture) start/stop should be no-ops, not raise."""
    tap = KeyTap()
    tap.start()
    # Sleep briefly so the thread can hit its tty setup and silently fail.
    time.sleep(0.05)
    tap.stop()
    assert tap.running is False


def test_keytap_get_pending_count_atomic_reset():
    """get_pending_count returns the value and resets to 0 atomically."""
    tap = KeyTap()
    with tap.lock:
        tap.pending_new_sessions = 5
    assert tap.get_pending_count() == 5
    assert tap.get_pending_count() == 0


def test_keytap_concurrent_increments_safe():
    """Lock should serialize concurrent increments correctly."""
    tap = KeyTap()

    def bump():
        for _ in range(100):
            with tap.lock:
                tap.pending_new_sessions += 1

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert tap.get_pending_count() == 400
