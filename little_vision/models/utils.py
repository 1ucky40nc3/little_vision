from typing import Any

import jax


def tree_equal(a: Any, b: Any) -> bool:
    a_values, a_treedef = jax.tree_util.tree_flatten(a)
    b_values, b_treedef = jax.tree_util.tree_flatten(b)
    
    return (
        a_treedef == b_treedef
        and all([
            (i == j).all() 
            for i, j in zip(a_values, b_values)
        ])
    )