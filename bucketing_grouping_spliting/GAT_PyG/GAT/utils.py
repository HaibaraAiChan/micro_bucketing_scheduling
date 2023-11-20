import dgl.backend as F

def expand_as_pair(input_, g=None):
    """Return a pair of same element if the input is not a pair.

    If the graph is a block, obtain the feature of destination nodes from the source nodes.

    Parameters
    ----------
    input_ : Tensor, dict[str, Tensor], or their pairs
        The input features
    g : DGLGraph or None
        The graph.

        If None, skip checking if the graph is a block.

    Returns
    -------
    tuple[Tensor, Tensor] or tuple[dict[str, Tensor], dict[str, Tensor]]
        The features for input and output nodes
    """
    if isinstance(input_, tuple):
        return input_
    elif g is not None and g.is_block:
        if isinstance(input_, Mapping):
            input_dst = {
                k: F.narrow_row(v, 0, g.number_of_dst_nodes(k))
                for k, v in input_.items()
            }
        else:
            input_dst = F.narrow_row(input_, 0, g.number_of_dst_nodes())
        return input_, input_dst
    else:
        return input_, input_
    
def get_nested_tuple_dimensions(t):
    if isinstance(t, tuple):
        print('t ',t)
        print(t[0].size())
        print(t[1].size())
        return [2]+ list(t[0].size())[1:]
