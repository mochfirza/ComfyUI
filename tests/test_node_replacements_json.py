"""Tests for auto-registration of node_replacements.json from custom node directories."""
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

# We can't import nodes.py directly (torch dependency), so we test the
# load_node_replacements_json logic by re-creating it from the same source.
# This validates the JSON parsing and NodeReplace construction logic.


class MockNodeReplace:
    """Mirrors comfy_api.latest._io.NodeReplace for testing."""
    def __init__(self, new_node_id, old_node_id, old_widget_ids=None,
                 input_mapping=None, output_mapping=None):
        self.new_node_id = new_node_id
        self.old_node_id = old_node_id
        self.old_widget_ids = old_widget_ids
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping


def load_node_replacements_json(module_dir, module_name, manager, NodeReplace=MockNodeReplace):
    """Standalone version of the function from nodes.py for testing."""
    import logging
    replacements_path = os.path.join(module_dir, "node_replacements.json")
    if not os.path.isfile(replacements_path):
        return

    try:
        with open(replacements_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            logging.warning(f"node_replacements.json in {module_name} must be a JSON object, skipping.")
            return

        count = 0
        for old_node_id, replacements in data.items():
            if not isinstance(replacements, list):
                logging.warning(f"node_replacements.json in {module_name}: value for '{old_node_id}' must be a list, skipping.")
                continue
            for entry in replacements:
                if not isinstance(entry, dict):
                    continue
                manager.register(NodeReplace(
                    new_node_id=entry.get("new_node_id", ""),
                    old_node_id=entry.get("old_node_id", old_node_id),
                    old_widget_ids=entry.get("old_widget_ids"),
                    input_mapping=entry.get("input_mapping"),
                    output_mapping=entry.get("output_mapping"),
                ))
                count += 1

        if count > 0:
            logging.info(f"Loaded {count} node replacement(s) from {module_name}/node_replacements.json")
    except json.JSONDecodeError as e:
        logging.warning(f"Failed to parse node_replacements.json in {module_name}: {e}")
    except Exception as e:
        logging.warning(f"Failed to load node_replacements.json from {module_name}: {e}")


class TestLoadNodeReplacementsJson(unittest.TestCase):
    """Test auto-registration of node_replacements.json from custom node directories."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mock_manager = MagicMock()

    def _write_json(self, data):
        path = os.path.join(self.tmpdir, "node_replacements.json")
        with open(path, "w") as f:
            json.dump(data, f)

    def _load(self):
        load_node_replacements_json(self.tmpdir, "test-node-pack", self.mock_manager)

    def test_no_file_does_nothing(self):
        """No node_replacements.json — should silently do nothing."""
        self._load()
        self.mock_manager.register.assert_not_called()

    def test_empty_object(self):
        """Empty {} — should do nothing."""
        self._write_json({})
        self._load()
        self.mock_manager.register.assert_not_called()

    def test_single_replacement(self):
        """Single replacement entry registers correctly."""
        self._write_json({
            "OldNode": [{
                "new_node_id": "NewNode",
                "old_node_id": "OldNode",
                "input_mapping": [{"new_id": "model", "old_id": "ckpt_name"}],
                "output_mapping": [{"new_idx": 0, "old_idx": 0}],
            }]
        })
        self._load()
        self.mock_manager.register.assert_called_once()
        registered = self.mock_manager.register.call_args[0][0]
        self.assertEqual(registered.new_node_id, "NewNode")
        self.assertEqual(registered.old_node_id, "OldNode")
        self.assertEqual(registered.input_mapping, [{"new_id": "model", "old_id": "ckpt_name"}])
        self.assertEqual(registered.output_mapping, [{"new_idx": 0, "old_idx": 0}])

    def test_multiple_replacements(self):
        """Multiple old_node_ids each with entries."""
        self._write_json({
            "NodeA": [{"new_node_id": "NodeB", "old_node_id": "NodeA"}],
            "NodeC": [{"new_node_id": "NodeD", "old_node_id": "NodeC"}],
        })
        self._load()
        self.assertEqual(self.mock_manager.register.call_count, 2)

    def test_multiple_alternatives_for_same_node(self):
        """Multiple replacement options for the same old node."""
        self._write_json({
            "OldNode": [
                {"new_node_id": "AltA", "old_node_id": "OldNode"},
                {"new_node_id": "AltB", "old_node_id": "OldNode"},
            ]
        })
        self._load()
        self.assertEqual(self.mock_manager.register.call_count, 2)

    def test_null_mappings(self):
        """Null input/output mappings (trivial replacement)."""
        self._write_json({
            "OldNode": [{
                "new_node_id": "NewNode",
                "old_node_id": "OldNode",
                "input_mapping": None,
                "output_mapping": None,
            }]
        })
        self._load()
        registered = self.mock_manager.register.call_args[0][0]
        self.assertIsNone(registered.input_mapping)
        self.assertIsNone(registered.output_mapping)

    def test_old_node_id_defaults_to_key(self):
        """If old_node_id is missing from entry, uses the dict key."""
        self._write_json({
            "OldNode": [{"new_node_id": "NewNode"}]
        })
        self._load()
        registered = self.mock_manager.register.call_args[0][0]
        self.assertEqual(registered.old_node_id, "OldNode")

    def test_invalid_json_skips(self):
        """Invalid JSON file — should warn and skip, not crash."""
        path = os.path.join(self.tmpdir, "node_replacements.json")
        with open(path, "w") as f:
            f.write("{invalid json")
        self._load()
        self.mock_manager.register.assert_not_called()

    def test_non_object_json_skips(self):
        """JSON array instead of object — should warn and skip."""
        self._write_json([1, 2, 3])
        self._load()
        self.mock_manager.register.assert_not_called()

    def test_non_list_value_skips(self):
        """Value is not a list — should warn and skip that key."""
        self._write_json({
            "OldNode": "not a list",
            "GoodNode": [{"new_node_id": "NewNode", "old_node_id": "GoodNode"}],
        })
        self._load()
        self.assertEqual(self.mock_manager.register.call_count, 1)

    def test_with_old_widget_ids(self):
        """old_widget_ids are passed through."""
        self._write_json({
            "OldNode": [{
                "new_node_id": "NewNode",
                "old_node_id": "OldNode",
                "old_widget_ids": ["width", "height"],
            }]
        })
        self._load()
        registered = self.mock_manager.register.call_args[0][0]
        self.assertEqual(registered.old_widget_ids, ["width", "height"])

    def test_set_value_in_input_mapping(self):
        """input_mapping with set_value entries."""
        self._write_json({
            "OldNode": [{
                "new_node_id": "NewNode",
                "old_node_id": "OldNode",
                "input_mapping": [
                    {"new_id": "method", "set_value": "lanczos"},
                    {"new_id": "size", "old_id": "dimension"},
                ],
            }]
        })
        self._load()
        registered = self.mock_manager.register.call_args[0][0]
        self.assertEqual(len(registered.input_mapping), 2)
        self.assertEqual(registered.input_mapping[0]["set_value"], "lanczos")
        self.assertEqual(registered.input_mapping[1]["old_id"], "dimension")

    def test_non_dict_entry_skipped(self):
        """Non-dict entries in the list are silently skipped."""
        self._write_json({
            "OldNode": [
                "not a dict",
                {"new_node_id": "NewNode", "old_node_id": "OldNode"},
            ]
        })
        self._load()
        self.assertEqual(self.mock_manager.register.call_count, 1)


if __name__ == "__main__":
    unittest.main()
