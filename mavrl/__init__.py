"""MAVRL: memory-augmented, varying-speed RL for a vision-only obstacle course.

Layered so the geometry and gate logic import cleanly without mujoco/torch --
that is what makes them testable on a machine with no simulator installed.

    course_gates  pure geometry: gate planes, traversal outcomes, gate ordering
    course_world  procedural course XML, layout sampling, semantic class map

Everything above those (env, SeVAE, memory, training) pulls in mujoco or torch.
"""

from mavrl.course_gates import (  # noqa: F401
    BarGate, BarSide, GateSequence, Opening, Outcome, WallGate,
)
from mavrl.course_world import (  # noqa: F401
    CourseLayout, SemClass, StationSpec, StationType,
    build_gates, sample_layout, sample_layout_for_stage,
)

__all__ = [
    "BarGate", "BarSide", "GateSequence", "Opening", "Outcome", "WallGate",
    "CourseLayout", "SemClass", "StationSpec", "StationType",
    "build_gates", "sample_layout", "sample_layout_for_stage",
]
