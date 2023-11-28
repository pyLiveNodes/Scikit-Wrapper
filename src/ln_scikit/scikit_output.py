from livenodes import Node
from livenodes_core_nodes.ports import Ports_empty, Ports_any
import multiprocessing as mp

class Scikit_output(Node):
    """Saves all incoming data in a queue and provides one-time cross-process access to it by converting the queue into a list.
    """
    ports_in = Ports_any()
    ports_out = Ports_empty()

    example_init = {'name': 'Scikit Output'}

    def __init__(self, name="Scikit Output", **kwargs):
        super().__init__(name, **kwargs)
        self.out = mp.SimpleQueue()

    def process(self, any, **kwargs):
        self.out.put(any)

    def get_state(self):
        res = []
        while not self.out.empty():
            res.append(self.out.get())
        return res