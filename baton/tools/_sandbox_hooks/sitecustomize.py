"""
Baton Sandbox Site Hook.
Automatically imported by Python when running code inside the Baton sandbox.
Ensures headless GUI execution (Pygame, Tkinter) executes safely without blocking for timeouts.
"""
import os
import sys
import threading
import time

if os.getenv("BATON_HEADLESS_SMOKE_TEST") == "1":
    def _smoke_test_guard():
        # Allow sufficient time for imports, window init, asset loading, and initial frame loop
        time.sleep(1.2)

        # 1. Pygame Auto-Quit Guard
        if "pygame" in sys.modules:
            pg = sys.modules["pygame"]
            if hasattr(pg, "get_init") and pg.get_init():
                try:
                    # Post standard pygame.QUIT event into the event queue
                    pg.event.post(pg.event.Event(pg.QUIT))
                except Exception:
                    pass
                time.sleep(1.0)
                # If still running after 1.0s, the event loop has verified healthy initialization
                if hasattr(pg, "get_init") and pg.get_init():
                    os._exit(0)

        # 2. Tkinter Auto-Quit Guard
        if "tkinter" in sys.modules:
            tk = sys.modules["tkinter"]
            try:
                if hasattr(tk, "_default_root") and tk._default_root:
                    tk._default_root.quit()
                    time.sleep(0.5)
            except Exception:
                pass

    t = threading.Thread(target=_smoke_test_guard, daemon=True)
    t.start()
