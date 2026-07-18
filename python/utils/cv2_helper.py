# -*- coding: utf-8 -*-
"""OpenCV GUI helpers for reliable window teardown."""
import cv2


def destroy_cv_windows(window_name=None, wait_ms=50, iterations=10):
    """Destroy OpenCV windows and pump the event loop.

    macOS/Cocoa needs a few waitKey pumps after destroyAllWindows for the
    windows to actually disappear.  This helper calls ``destroyAllWindows``,
    pumps the event loop, and then destroys again for good measure.

    Args:
        window_name: Optional single window name to destroy.  If None, all
            windows are destroyed.
        wait_ms: Milliseconds to wait per pump iteration.
        iterations: Number of pump iterations.
    """
    try:
        if window_name is not None:
            cv2.destroyWindow(window_name)
        else:
            cv2.destroyAllWindows()

        for _ in range(iterations):
            cv2.waitKey(wait_ms)

        if window_name is not None:
            cv2.destroyWindow(window_name)
        else:
            cv2.destroyAllWindows()
    except Exception:
        pass
