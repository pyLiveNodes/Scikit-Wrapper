from sklearn.base import BaseEstimator, ClassifierMixin
from livenodes import Node, Graph


class LN_Estimator (BaseEstimator, ClassifierMixin):
    def __init__(self, fit_graph, prd_graph, fit_x_channel, fit_y_channel, fit_model_channel, prd_x_channel, prd_y_channel, prd_model_channel, fit_params={}, prd_params={}):
        # Store pipelines, these should be dicts
        self.fit_graph_dct = fit_graph
        self.prd_graph_dct = prd_graph
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

    @classmethod
    def construct_fit_graph(cls, fit_graph_dct, fit_x_channel, fit_y_channel, fit_model_channel, fit_params, X, y):
        # Construct IO dict def:
        fit_io = {
            f'fit x [Scikit Input]': dict(value=X),
            f'fit y [Scikit Input]': dict(value=y),
            f'fit model [Scikit Output]': dict(),
        }

        # Apply params and add I/O Nodes
        fit_graph_dct['Nodes'] = {**fit_graph_dct['Nodes'], **fit_params, **fit_io}
        
        # Add I/O Connections
        fit_graph_dct['Inputs'] += [
            f"fit x [Scikit Input].any -> {fit_x_channel}",
            f"fit y [Scikit Input].any -> {fit_y_channel}",
            f"{fit_model_channel} -> fit model [Scikit Output].any",
        ]

        # Construct graph
        graph = Node.from_compact_dict(fit_graph_dct)

        # return graph as well as model node
        return graph, cls._get_node(graph, 'fit model [Scikit Output]')


    @classmethod
    def construct_prd_graph(cls, prd_graph_dct, prd_x_channel, prd_y_channel, prd_model_channel, prd_params, X, model):
        # Construct IO dict def:
        prd_io = {
            f'prd x [Scikit Input]': dict(value=X),
            f'prd model [Scikit Input]': dict(value=[model]),
            f'prd y [Scikit Output]': dict(),
        }

        # Apply params and add I/O Nodes
        prd_graph_dct['Nodes'] = {**prd_graph_dct['Nodes'], **prd_params, **prd_io}
        
        # Add I/O Connections
        prd_graph_dct['Inputs'] += [
            f"prd x [Scikit Input].any -> {prd_x_channel}",
            f"prd model [Scikit Input].any -> {prd_model_channel}",
            f"{prd_y_channel} -> prd y [Scikit Output].any",
        ]

        # Construct graph
        graph = Node.from_compact_dict(prd_graph_dct)

        # return graph as well as model node
        return graph, cls._get_node(graph, 'prd y [Scikit Output')

    
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

        self.model = model_node.get_state()[0]
        return self
        
    def predict(self, X):
        initial_node, prd_node = self.construct_prd_graph(self.prd_graph_dct, self.prd_x_channel, self.prd_y_channel, self.prd_model_channel, self.prd_params, X, self.model)

        g = Graph(start_node=initial_node)
        g.start_all()
        g.join_all()
        g.stop_all()

        return prd_node.get_state()
