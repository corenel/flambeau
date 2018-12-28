import unittest

from flambeau.misc import config


class TestConfig(unittest.TestCase):
    def test_json_profile(self):
        hps = config.load_profile('config.json')
        config.dump_profile(hps, output_dir='/tmp', file_type='json')
        hps_ = config.load_profile('/tmp/config.json')
        self.assertDictEqual(hps, hps_)

    def test_yaml_profile(self):
        hps = config.load_profile('config.yaml')
        config.dump_profile(hps, output_dir='/tmp', file_type='yaml')
        hps_ = config.load_profile('/tmp/config.yaml')
        self.assertDictEqual(hps, hps_)


if __name__ == '__main__':
    unittest.main()
