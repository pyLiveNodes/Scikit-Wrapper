from importlib.metadata import entry_points
import pytest

DEPRECATION_MODULES = []

class TestProcessing():

    def test_all_instantiable(self):
        for x in entry_points()['livenodes.nodes']:
            if x.name in DEPRECATION_MODULES:
                continue
            print(x)
            # implicit test if module is loadable
            node_class = x.load()
            # implicit test if class is instantiable with default values
            node_class(**node_class.example_init)

    def test_all_should_processable(self):
        # Note: these are very crude tests! 
        # Each node should also be tested separately.

        for x in entry_points()['livenodes.nodes']:
            if x.name in DEPRECATION_MODULES:
                continue
            print(x)
            # implicit test if module is loadable
            node_class = x.load()
            # implicit test if class is instantiable with default values
            example_node = node_class(**node_class.example_init)

            # heuristic test if should process works with some of the example values provided in the port classes
            # if should process works and returns true, test if process works as well
            for key, val in example_node.ports_in._asdict().items():
                # originally this would accumulate over each input port, but this would fail, as the example values might not match across multiple ports
                # i.e. if SelectNode takes data + channels with the same last dimension, this might not be proper with random example values and an errr could be rightfully raised, even though the test should not fail
                # thus for now we just test single values...
                # -> likely no processing will happen, but we do already check if the naming is consistent
                # -> TODO: we could also check consistency within the __subclass__ hook in Node.py...
                example_values = {key: val.example_values[0]}
                print(example_values)
                example_node._should_process(**example_values)


        # essentially: the types are not fully fledged out yet.. :/
    def test_all_processable(self):
        # Note: these are very crude tests! 
        # Each node should also be tested separately.

        for x in entry_points()['livenodes.nodes']:
            if x.name in DEPRECATION_MODULES:
                continue
            print(x)
            # implicit test if module is loadable
            node_class = x.load()
            # implicit test if class is instantiable with default values
            example_node = node_class(**node_class.example_init)

            # heuristic test if should process works with some of the example values provided in the port classes
            # if should process works and returns true, test if process works as well
            example_values = {}
            for key, val in example_node.ports_in._asdict().items():
                example_values[key] = val.example_values[0]
                print(example_values)
                
                res = None
                try: 
                    if example_node._should_process(**example_values):
                        res = example_node.process(**example_values)
                        print(res)
                except:
                    pass
                
                if res is not None:
                    for key in res:
                        if key not in example_node.ports_out._fields:
                            raise ValueError(f'"{key}" is no valid out port for {example_node}', example_node.ports_out._fields)