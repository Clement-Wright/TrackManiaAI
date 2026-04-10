from __future__ import annotations

import numpy as np

from tm20ai.control.gamepad import GamepadController


class FakePad:
    def __init__(self) -> None:
        self.left_trigger_value = None
        self.right_trigger_value = None
        self.left_stick = None
        self.update_calls = 0

    def left_trigger(self, value: int) -> None:
        self.left_trigger_value = value

    def right_trigger(self, value: int) -> None:
        self.right_trigger_value = value

    def left_joystick(self, x_value: int, y_value: int) -> None:
        self.left_stick = (x_value, y_value)

    def update(self) -> None:
        self.update_calls += 1


def test_gamepad_clamps_and_maps_action() -> None:
    fake = FakePad()
    controller = GamepadController(backend=fake)

    applied = controller.apply([1.4, -0.2, -2.0])

    assert np.allclose(applied, np.asarray([1.0, 0.0, -1.0], dtype=np.float32))
    assert fake.right_trigger_value == 255
    assert fake.left_trigger_value == 0
    assert fake.left_stick == (-32768, 0)
    assert fake.update_calls == 1


def test_gamepad_neutral_action_is_zeroed() -> None:
    neutral = GamepadController.neutral_action()
    assert neutral.dtype == np.float32
    assert neutral.tolist() == [0.0, 0.0, 0.0]
