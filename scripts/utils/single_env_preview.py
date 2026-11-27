"""Render one training environment without changing any physics or PPO state."""

from contextlib import contextmanager


def configure_preview(cfg, env_index: int, render_interval: int, width: int, height: int) -> None:
    """Configure a small viewport that follows one robot at a lower render rate."""
    if not 0 <= env_index < cfg.scene.num_envs:
        raise ValueError(f"Preview environment {env_index} outside [0, {cfg.scene.num_envs})")
    if render_interval < cfg.decimation or width <= 0 or height <= 0:
        raise ValueError("Preview render interval must be >= decimation; dimensions must be positive")
    cfg.sim.render_interval = render_interval
    cfg.ui_window_class_type = None
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.env_index = env_index
    cfg.viewer.eye = (2.0, 2.0, 1.2)
    cfg.viewer.lookat = (0.0, 0.0, 0.0)
    cfg.viewer.resolution = (width, height)
    # Record preview provenance alongside the normal training config.
    cfg.single_env_preview = {"env_index": env_index, "width": width, "height": height}


def hide_other_environments(stage, num_envs: int, env_index: int) -> int:
    """Author visibility before stage attachment / physics initialization."""
    from pxr import Sdf, UsdGeom

    env_root = stage.GetPrimAtPath("/World/envs")
    if not env_root.IsValid():
        raise RuntimeError("Single environment preview requires /World/envs")
    hidden = 0
    with Sdf.ChangeBlock():
        for prim in env_root.GetChildren():
            if prim.GetName() == f"env_{env_index}":
                UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
                continue
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            hidden += 1
    if hidden != num_envs - 1:
        raise RuntimeError(f"Preview expected {num_envs - 1} hidden environments, got {hidden}")
    return hidden


@contextmanager
def preview_initialization():
    """Pause viewport updates while a large physics scene is being constructed."""
    import carb
    from omni.kit.viewport.utility import get_active_viewport

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Single environment preview requires an active GUI viewport")
    settings = carb.settings.get_settings()
    reason = "/app/hangDetector/disableReasons/SingleEnvironmentPreviewInitialization"
    was_enabled = viewport.updates_enabled
    viewport.updates_enabled = False
    # Kit's default 120s detector mistook the 20000-env cloning call for a hang.
    # Suspend only during construction; restore detection for actual training.
    settings.set(reason, "1")
    print("[Preview] Building physics scene with viewport paused; large environment counts can take minutes.", flush=True)
    try:
        yield
    finally:
        settings.destroy_item(reason)
        viewport.updates_enabled = was_enabled


def isolate_preview(env, env_index: int, width: int, height: int) -> None:
    """Finish GUI setup after early visibility overrides have been applied."""
    import omni.appwindow
    import omni.ui as ui
    from omni.kit.viewport.utility import get_active_viewport

    hidden = getattr(env, "_preview_hidden_env_count", None)
    if hidden is None:
        raise RuntimeError("This environment does not implement preview hiding before scene attachment")
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Single environment preview requires an active GUI viewport")
    viewport.set_texture_resolution((width, height))
    # A saved desktop layout can otherwise restore a maximized window and
    # override SimulationApp's startup window dimensions.
    app_window = omni.appwindow.get_default_app_window()
    app_window.restore_window()
    app_window.resize(width, height)
    # Keep the viewport and the normal application menu; hide docked editors.
    for window in ui.Workspace.get_windows():
        if "viewport" not in window.title.lower():
            window.visible = False

    print(
        f"[Preview] env_{env_index} visible; {hidden} environments hidden visually, "
        f"all {env.num_envs} environments still simulated. "
        f"Viewport={width}x{height}; camera follows robot; "
        f"render every {env.cfg.sim.render_interval} physics steps.",
        flush=True,
    )
