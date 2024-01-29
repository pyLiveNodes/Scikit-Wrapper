from livenodes import Producer
from livenodes_core_nodes.ports import Ports_empty, Ports_any

class Scikit_input(Producer):
    """Feeds all initially set data one by one into the pipeline.
    """
    ports_in = Ports_empty()
    ports_out = Ports_any()

    example_init = {'name': 'Scikit Input', 'values': [1]}

    def __init__(self, name="Scikit Input", values=[1], **kwargs):
        super().__init__(name=name, **kwargs)
        self.values = values

    def _settings(self):
        return {\
            "values": self.values,
           }

    def _run(self):
        for val in self.values:
            yield self.ret(any=val)
