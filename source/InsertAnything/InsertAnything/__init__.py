# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""InsertAnything Isaac Lab extension."""

# Register Gym environments.
from . import tasks as _tasks

_REGISTERED_TASK_MODULES = (_tasks,)
