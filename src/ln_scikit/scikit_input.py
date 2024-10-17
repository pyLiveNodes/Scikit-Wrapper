from livenodes import Producer
from ln_ports import Ports_empty, Ports_collection, Port_Any, Port_Number

class Ports_any(Ports_collection):
    any: Port_Any = Port_Any("Any")
    percent: Port_Number = Port_Number("Percent")

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
        v = list(self.values)
        l = len(v)
        for i, val in enumerate(v):
            yield self.ret(any=val, percent=(i+1)/l)
