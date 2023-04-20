from functools import partial
from beachlanding.combat_env_vs_bot import CombatEnv
from beachlanding.multiagentenv import MultiAgentEnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {"blv4": partial(env_fn, env=CombatEnv)}
