
import unittest
from unittest.mock import patch
import sys
import os

class TestConfig(unittest.TestCase):

    def setUp(self):
        # Save and remove src.config so each test reimports it
        self._saved_config = sys.modules.get('src.config')
        if 'src.config' in sys.modules:
            del sys.modules['src.config']

    def tearDown(self):
        # Restore the original config module so other tests are unaffected
        if self._saved_config is not None:
            sys.modules['src.config'] = self._saved_config

    @patch('dotenv.load_dotenv')
    def test_addon_environment_config(self, mock_load_dotenv):
        """Test that config uses addon paths when SUPERVISOR_TOKEN is set."""
        with patch.dict(os.environ, {'SUPERVISOR_TOKEN': 'dummy-token'}, clear=True):
            from src import config
            self.assertEqual(config.UNIFIED_STATE_FILE, "/opt/ml_heating/unified_thermal_state.json")

    @patch('dotenv.load_dotenv')
    def test_standalone_environment_config(self, mock_load_dotenv):
        """Test that config uses standalone paths when SUPERVISOR_TOKEN is not set."""
        with patch.dict(os.environ, {}, clear=True):
            from src import config
            self.assertEqual(config.UNIFIED_STATE_FILE, "/opt/ml_heating/unified_thermal_state.json")


if __name__ == '__main__':
    unittest.main()
