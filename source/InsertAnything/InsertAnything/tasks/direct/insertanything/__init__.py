import gymnasium as gym

from . import agents
from .insertanything_env_cfg import (
    InsertAnythingCircleHole_I_Cfg,
    InsertAnythingHexagonHole_III_Cfg,
    InsertAnythingHexagonHole_IV_Cfg,
    InsertAnythingLHole_III_Cfg,
    InsertAnythingSquareHole_II_Cfg,
    InsertAnythingTriangleHole_IV_Cfg,
)


def _register(task_id: str, env_cfg_entry_point, rl_games_cfg: str):
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.insertanything_env:InsertAnythingEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "rl_games_cfg_entry_point": f"{agents.__name__}:{rl_games_cfg}",
        },
    )


_register(
    "InsertAnything-CircleHole-I-Direct-v0",
    InsertAnythingCircleHole_I_Cfg,
    "rl_games_ppo_circle_cfg.yaml",
)
_register(
    "InsertAnything-SquareHole-II-Direct-v0",
    InsertAnythingSquareHole_II_Cfg,
    "rl_games_ppo_square_cfg.yaml",
)
_register(
    "InsertAnything-LHole-III-Direct-v0",
    InsertAnythingLHole_III_Cfg,
    "rl_games_ppo_L_cfg.yaml",
)
_register(
    "InsertAnything-TriangleHole-IV-Direct-v0",
    InsertAnythingTriangleHole_IV_Cfg,
    "rl_games_ppo_triangle_cfg.yaml",
)
_register(
    "InsertAnything-HexagonHole-III-Direct-v0",
    InsertAnythingHexagonHole_III_Cfg,
    "rl_games_ppo_hexagon_cfg.yaml",
)
_register(
    "InsertAnything-HexagonHole-IV-Direct-v0",
    InsertAnythingHexagonHole_IV_Cfg,
    "rl_games_ppo_hexagon_cfg.yaml",
)
