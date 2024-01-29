from functools import reduce
from sklearn.base import BaseEstimator, ClassifierMixin
from livenodes import Node, Graph

def flatten_reduce_lambda(matrix):
    return list(reduce(lambda x, y: x + list(y), matrix, []))

def merge_dicts_deep(*dicts):
    """
    Merge multiple dicts into one dict
    """
    res = {}
    for dct in dicts:
        for k, v in dct.items():
            if k in res and isinstance(res[k], dict) and isinstance(v, dict):
                res[k] = merge_dicts_deep(res[k], v)
            else:
                res[k] = v
    return res

class LN_Estimator (BaseEstimator, ClassifierMixin):
    def __init__(self, fit_graph_dct, prd_graph_dct, fit_x_channel, fit_y_channel, fit_model_channel, prd_x_channel, prd_y_channel, prd_model_channel, fit_params={}, prd_params={}):
        # Store pipelines, these should be dicts
        self.fit_graph_dct = fit_graph_dct
        self.prd_graph_dct = prd_graph_dct
        # Store channels, these should be of format "<Node Name> [<Node Class>].<node_channel>"
        self.fit_x_channel = fit_x_channel
        self.fit_y_channel = fit_y_channel
        self.fit_model_channel = fit_model_channel
        self.prd_x_channel = prd_x_channel
        self.prd_y_channel = prd_y_channel
        self.prd_model_channel = prd_model_channel
        # Store params, these should be dicts of format {"<Node Name> [<Node Class>]": {<Node Settings>}}
        self.fit_params = fit_params
        self.prd_params = prd_params

    @staticmethod
    def construct_fit_graph(fit_graph_dct, fit_x_channel, fit_y_channel, fit_model_channel, fit_params, X, y):
        # Construct IO dict def:
        fit_io = {
            f'fit x [Scikit_input]': dict(values=X, name='fit x'),
            f'fit y [Scikit_input]': dict(values=y, name='fit y'),
            f'fit model [Scikit_output]': dict(name='fit model'),
        }

        # Apply params and add I/O Nodes
        nodes = merge_dicts_deep(fit_graph_dct['Nodes'], fit_params, fit_io)
        
        # Add I/O Connections
        inputs = fit_graph_dct['Inputs'] + [
            f"fit x [Scikit_input].any -> {fit_x_channel}",
            f"fit y [Scikit_input].any -> {fit_y_channel}",
            f"{fit_model_channel} -> fit model [Scikit_output].any",
        ]

        # Construct graph
        graph = Node.from_compact_dict(dict(Nodes=nodes, Inputs=inputs))

        # return graph as well as model node
        return graph, 'fit model [Scikit_output]'


    @staticmethod
    def construct_prd_graph(prd_graph_dct, prd_x_channel, prd_y_channel, prd_model_channel, prd_params, X, model):
        # Construct IO dict def:
        prd_io = {
            f'prd x [Scikit_input]': dict(values=X, name='prd x'),
            f'prd model [Scikit_input]': dict(values=[model], name='prd model'),
            f'prd y [Scikit_output]': dict(name='prd y'),
        }

        # Apply params and add I/O Nodes
        nodes = merge_dicts_deep(prd_graph_dct['Nodes'], prd_params, prd_io)
        
        # Add I/O Connections
        inputs = prd_graph_dct['Inputs'] + [
            f"prd x [Scikit_input].any -> {prd_x_channel}",
            f"prd model [Scikit_input].any -> {prd_model_channel}",
            f"{prd_y_channel} -> prd y [Scikit_output].any",
        ]

        # Construct graph
        graph = Node.from_compact_dict(dict(Nodes=nodes, Inputs=inputs))

        # return graph as well as model node
        return graph, 'prd y [Scikit_output]'

    
    @staticmethod
    def _get_node(graph, node_name):
        for x in graph.nodes:
            if str(x) == node_name:
                return x
        
    def fit(self, X, y):
        initial_node, model_node = self.construct_fit_graph(self.fit_graph_dct, self.fit_x_channel, self.fit_y_channel, self.fit_model_channel, self.fit_params, X, y)

        g = Graph(start_node=initial_node)
        g.start_all()
        g.join_all()
        g.stop_all()

        self.model = self._get_node(g, model_node).get_state()[-1]
        return self
        
    def predict(self, X):
        initial_node, prd_node = self.construct_prd_graph(self.prd_graph_dct, self.prd_x_channel, self.prd_y_channel, self.prd_model_channel, self.prd_params, X, self.model)

        g = Graph(start_node=initial_node)
        g.start_all()
        g.join_all()
        g.stop_all()

        prediction = self._get_node(g, prd_node).get_state()
        return flatten_reduce_lambda(prediction)
