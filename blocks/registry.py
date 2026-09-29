# Mahdi Abdollahpour
# mahdi.abdollahpour@unibo.it
# 2026

"""Shared block registry for graph-constructible layers."""

BLOCK_REGISTRY = {}
BLOCK_ALIASES = {}
BLOCK_NEEDS = {}


def register_block(cls=None, *, name=None, needs=()):
    """Register a block class under its class name or a custom name.

    ``needs`` declares the runtime objects the block requires at construction
    time (e.g. ``needs=('sys',)``). The graph builder injects them into the
    block's ``__init__`` kwargs, so the config file does not have to restate
    values that already live in the system parameters.
    """

    def _register(target_cls):
        key = name or target_cls.__name__
        BLOCK_REGISTRY[key] = target_cls
        if needs:
            BLOCK_NEEDS[key] = tuple(needs)
        return target_cls

    if cls is None:
        return _register
    return _register(cls)


def register_alias(alias, target):
    """Register an alias that resolves to an existing block name or class."""

    BLOCK_ALIASES[alias] = target


def get_block_registry(include_aliases=False):
    """Return a copy of the shared block registry."""

    registry = dict(BLOCK_REGISTRY)
    if include_aliases:
        for alias, target in BLOCK_ALIASES.items():
            registry[alias] = BLOCK_REGISTRY[target] if isinstance(target, str) else target
    return registry


def get_block_needs(include_aliases=False):
    """Return a copy of the init-time dependency map: name -> tuple of deps."""

    needs = dict(BLOCK_NEEDS)
    if include_aliases:
        for alias, target in BLOCK_ALIASES.items():
            key = target if isinstance(target, str) else target.__name__
            if key in BLOCK_NEEDS:
                needs[alias] = BLOCK_NEEDS[key]
    return needs
