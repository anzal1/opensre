"""Safe-boundary handling for in-flight session-goal controls."""

from __future__ import annotations

from core.agent_harness.spi.cancel import HostCancelReason
from core.agent_harness.spi.session_goal import (
    clear_session_goal,
    pause_active_session_goal,
)
from core.agent_harness.spi.session_state import clear_pending_autosubmit, session_terminal
from surfaces.interactive_shell.session import Session


def apply_goal_control(session: Session, reason: HostCancelReason) -> bool:
    """Apply ``reason`` after the active worker releases session ownership."""
    if reason is HostCancelReason.GOAL_PAUSE:
        return pause_active_session_goal(session) is not None
    if reason is HostCancelReason.GOAL_CLEAR:
        clear_pending_autosubmit(session)
        if getattr(session, "session_goal", None) is None:
            return False
        clear_session_goal(session)
        return True
    raise ValueError(f"Not a goal control reason: {reason}")


def mark_inflight_goal_control(session: Session, reason: HostCancelReason) -> None:
    """Remember that the queued slash command follows boundary handling."""
    terminal = session_terminal(session)
    if terminal is None:
        return
    key = reason.value
    terminal.pending_inflight_goal_controls[key] = (
        terminal.pending_inflight_goal_controls.get(key, 0) + 1
    )


def consume_inflight_goal_control(session: Session, reason: HostCancelReason) -> bool:
    """Consume one queued slash command already handled at the boundary."""
    terminal = session_terminal(session)
    if terminal is None:
        return False
    key = reason.value
    pending = terminal.pending_inflight_goal_controls.get(key, 0)
    if pending <= 0:
        return False
    if pending == 1:
        terminal.pending_inflight_goal_controls.pop(key)
    else:
        terminal.pending_inflight_goal_controls[key] = pending - 1
    return True


__all__ = [
    "apply_goal_control",
    "consume_inflight_goal_control",
    "mark_inflight_goal_control",
]
