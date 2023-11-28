import glob
from os.path import dirname, basename, isfile, join
from importlib.metadata import entry_points

import pytest


@pytest.fixture
def discovered_modules():
    exclude = ['__init__']
    modules = glob.glob(join(dirname(__file__), '../src/livenodes_ease_tsd/', "*.py"))
    names = [basename(f)[:-3] for f in modules if isfile(f)]
    return [f for f in names if not f in exclude]

class TestProcessing():
    def test_modules_discoverable(self, discovered_modules):
        assert len(discovered_modules) > 0

    def test_all_declared(self, discovered_modules):
        livnodes_entrypoints = [x.name for x in entry_points()['livenodes.nodes']]

        print(set(discovered_modules).difference(set(livnodes_entrypoints)))
        assert set(discovered_modules) <= set(livnodes_entrypoints)

    def test_loads_class(self):
        ease_read_plux = [x.load() for x in entry_points()['livenodes.nodes'] if x.name == 'ease_read_plux'][0]
        from livenodes_ease_tsd.ease_read_plux import EASE_read_plux
        assert EASE_read_plux == ease_read_plux

    def test_all_loadable(self):
        for x in entry_points()['livenodes.nodes']:
            x.load()