"""ProtoFlow utils test suite."""

import io
import unittest

from protoflow import utils


class TestUtils(unittest.TestCase):
    def setUp(self):
        @utils.memoize(verbose=True)
        def talking_parrot(x, talk_loud=False):
            x = str(x)
            if talk_loud:
                return x.upper()
            return x

        self.talking_parrot = talking_parrot

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_memoize_new_1(self, stdout):
        self.talking_parrot('Blubb!', talk_loud=False)
        expected = "Adding NEW rv talking_parrot('Blubb!',){'talk_loud': False} to cache.\n"
        self.assertEqual(expected, stdout.getvalue())

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_memoize_old_1(self, stdout):
        self.talking_parrot('Blubb!', talk_loud=True)
        expected = "Adding NEW rv talking_parrot('Blubb!',){'talk_loud': True} to cache.\n"
        self.assertEqual(expected, stdout.getvalue())
        self.talking_parrot('Blubb!', talk_loud=True)
        expected += "Using OLD rv talking_parrot('Blubb!',){'talk_loud': True} from cache.\n"
        self.assertEqual(expected, stdout.getvalue())
        self.talking_parrot('Blah!', talk_loud=True)
        expected += "Adding NEW rv talking_parrot('Blah!',){'talk_loud': True} to cache.\n"
        self.assertEqual(expected, stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
