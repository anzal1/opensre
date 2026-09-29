"""Final interactive-session persistence after runtime shutdown."""

from __future__ import annotations

import logging

from core.agent_harness import SessionManager
from core.agent_harness.spi.session_goal import apply_session_goal_control
from infrastructure.turn_host.session_lock import session_execution_lock
from surfaces.interactive_shell.runtime.core.state import ReplState
from surfaces.interactive_shell.session import Session

logger = logging.getLogger(__name__)


def close_repl_session(session: Session, state: ReplState) -> None:
    """Persist final state unless forced exit left a worker owning the session."""
    if state.has_detached_turn_worker():
        logger.warning(
            "Skipping final session close because detached turn work still owns session state"
        )
        return
    goal_control = state.requested_goal_control()
    manager = SessionManager.for_session(session)
    with session_execution_lock(session.session_id):
        manager.refresh_from_storage(session)
        if goal_control is not None:
            apply_session_goal_control(session, goal_control)
        manager.close(session)


__all__ = ["close_repl_session"]
