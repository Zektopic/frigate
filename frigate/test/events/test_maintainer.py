import unittest

from frigate.events.maintainer import should_update_db, should_update_state


class TestShouldUpdateDb(unittest.TestCase):
    def setUp(self):
        self.base_event = {
            "end_time": None,
            "has_clip": False,
            "has_snapshot": False,
            "top_score": 0.5,
            "entered_zones": [],
            "average_estimated_speed": None,
            "velocity_angle": None,
            "recognized_license_plate": None,
            "path_data": None,
        }

    def _get_events(self):
        return dict(self.base_event), dict(self.base_event)

    def test_event_ending_previously_saved_with_clip(self):
        prev_event, current_event = self._get_events()
        prev_event["has_clip"] = True
        current_event["has_clip"] = (
            True  # might be false if mid-event changed, but test anyway
        )
        current_event["end_time"] = 12345.0
        self.assertTrue(should_update_db(prev_event, current_event))

    def test_event_ending_previously_saved_with_snapshot(self):
        prev_event, current_event = self._get_events()
        prev_event["has_snapshot"] = True
        current_event["end_time"] = 12345.0
        self.assertTrue(should_update_db(prev_event, current_event))

    def test_event_ending_not_previously_saved(self):
        prev_event, current_event = self._get_events()
        current_event["end_time"] = 12345.0
        self.assertFalse(should_update_db(prev_event, current_event))

    def test_first_time_clip_turned_true(self):
        prev_event, current_event = self._get_events()
        current_event["has_clip"] = True
        self.assertTrue(should_update_db(prev_event, current_event))

    def test_first_time_snapshot_turned_true(self):
        prev_event, current_event = self._get_events()
        current_event["has_snapshot"] = True
        self.assertTrue(should_update_db(prev_event, current_event))

    def test_values_changed(self):
        fields_to_change = [
            ("top_score", 0.8),
            ("entered_zones", ["zone1"]),
            ("end_time", 12345.0),
            ("average_estimated_speed", 10.5),
            ("velocity_angle", 45.0),
            ("recognized_license_plate", "ABC-123"),
            ("path_data", "some_data"),
        ]

        for field, new_value in fields_to_change:
            with self.subTest(field=field):
                prev_event, current_event = self._get_events()
                # we must have clip or snapshot true for this check
                prev_event["has_clip"] = True
                current_event["has_clip"] = True

                # change the field
                current_event[field] = new_value
                self.assertTrue(should_update_db(prev_event, current_event))

    def test_no_changes_with_clip(self):
        prev_event, current_event = self._get_events()
        prev_event["has_clip"] = True
        current_event["has_clip"] = True
        self.assertFalse(should_update_db(prev_event, current_event))

    def test_mid_event_lost_clip_and_snapshot(self):
        prev_event, current_event = self._get_events()
        prev_event["has_clip"] = True
        current_event["has_clip"] = False
        current_event["has_snapshot"] = False
        current_event["end_time"] = 12345.0
        self.assertTrue(should_update_db(prev_event, current_event))


class TestShouldUpdateState(unittest.TestCase):
    def setUp(self):
        self.base_event = {
            "stationary": False,
            "attributes": [],
            "sub_label": None,
            "current_zones": [],
        }

    def _get_events(self):
        return dict(self.base_event), dict(self.base_event)

    def test_stationary_changed(self):
        prev_event, current_event = self._get_events()
        current_event["stationary"] = True
        self.assertTrue(should_update_state(prev_event, current_event))

    def test_attributes_changed(self):
        prev_event, current_event = self._get_events()
        current_event["attributes"] = [{"label": "face", "score": 0.9}]
        self.assertTrue(should_update_state(prev_event, current_event))

    def test_sub_label_changed(self):
        prev_event, current_event = self._get_events()
        current_event["sub_label"] = "person"
        self.assertTrue(should_update_state(prev_event, current_event))

    def test_current_zones_changed(self):
        prev_event, current_event = self._get_events()
        current_event["current_zones"] = ["front_yard"]
        self.assertTrue(should_update_state(prev_event, current_event))

    def test_current_zones_unordered_no_change(self):
        prev_event, current_event = self._get_events()
        prev_event["current_zones"] = ["zone1", "zone2"]
        current_event["current_zones"] = ["zone2", "zone1"]
        self.assertFalse(should_update_state(prev_event, current_event))

    def test_no_changes(self):
        prev_event, current_event = self._get_events()
        self.assertFalse(should_update_state(prev_event, current_event))


if __name__ == "__main__":
    unittest.main()
