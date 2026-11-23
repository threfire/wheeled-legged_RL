"""Environment wrapper for the converted dogleg asset."""

from .env import WheelbipeV14Env


class DoglegEnv(WheelbipeV14Env):
    """Reuse the wheelbipe locomotion environment with dogleg bookkeeping."""

    def _validate_v14_bookkeeping(self) -> None:
        checks = {
            "legs_act": (len(self._legs_act_idx), 4),
            "wheel": (len(self._wheel_idx), 2),
            "ordered_leg_joint_idx": (len(self._ordered_leg_joint_idx), 12),
            "ordered_leg_body_idx": (len(self._ordered_leg_body_idx), 12),
        }
        if self._use_gimbal:
            checks["gimbal_yaw"] = (len(self._gimbal_yaw_idx), 1)
            checks["gimbal_pitch"] = (len(self._gimbal_pitch_idx), 1)
        bad_checks = [
            f"{name}={actual} (expected {expected})"
            for name, (actual, expected) in checks.items()
            if actual != expected
        ]
        if bad_checks:
            raise RuntimeError("Dogleg asset discovery mismatch: " + ", ".join(bad_checks))

