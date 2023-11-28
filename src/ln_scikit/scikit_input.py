import asyncio
from livenodes.producer_async import Producer_async
from livenodes_core_nodes.ports import Ports_empty, Ports_any

class Scikit_input(Producer_async):
    """Feeds all initially set data one by one into the pipeline.
    """
    ports_in = Ports_empty()
    ports_out = Ports_any()

    example_init = {'name': 'Scikit Input', 'value': [1]}

    def __init__(self, name="Scikit Input", value=[1], **kwargs):
        super().__init__(name=name, **kwargs)
        self.value = value

    def _settings(self):
        return {\
            "value": self.value,
           }

    async def _async_run(self):
        for val in self.values:
            yield self.ret(any=val)
            await asyncio.sleep(0) # so other tasks can run
