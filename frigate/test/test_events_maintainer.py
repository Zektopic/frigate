import unittest
from typing import Any
from copy import deepcopy

from frigate.events.maintainer import should_update_state

class TestEventsMaintainer(unittest.TestCase):
    def setUp(self) -> None:
        self.base_event = {
            "stationary": False,
            "attributes": [],
            "sub_label": None,
            "current_zones": [],
        }

    def test_should_update_state_identical(self) -> None:
        prev_event = deepcopy(self.base_event)
        current_event = deepcopy(self.base_event)

        self.assertFalse(should_update_state(prev_event, current_event))

    def test_should_update_state_stationary_changed(self) -> None:
        prev_event = deepcopy(self.base_event)
        current_event = deepcopy(self.base_event)

        current_event["stationary"] = True

        self.assertTrue(should_update_state(prev_event, current_event))

    def test_should_update_state_attributes_changed(self) -> None:
        prev_event = deepcopy(self.base_event)
        current_event = deepcopy(self.base_event)

        current_event["attributes"] = [{"label": "face", "score": 0.8}]

        self.assertTrue(should_update_state(prev_event, current_event))

    def test_should_update_state_sub_label_changed(self) -> None:
        prev_event = deepcopy(self.base_event)
        current_event = deepcopy(self.base_event)

        current_event["sub_label"] = "dog"

        self.assertTrue(should_update_state(prev_event, current_event))

    def test_should_update_state_current_zones_changed(self) -> None:
        prev_event = deepcopy(self.base_event)
        current_event = deepcopy(self.base_event)

        current_event["current_zones"] = ["front_yard"]

        self.assertTrue(should_update_state(prev_event, current_event))

    def test_should_update_state_current_zones_changed_order(self) -> None:
        prev_event = deepcopy(self.base_event)
        current_event = deepcopy(self.base_event)

        prev_event["current_zones"] = ["zone1", "zone2"]
        current_event["current_zones"] = ["zone2", "zone1"]

        self.assertFalse(should_update_state(prev_event, current_event))
