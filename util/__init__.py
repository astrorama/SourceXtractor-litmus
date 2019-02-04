
def get_column(catalog, col):
    """
    Convenience function to allow accessing multidimensional cells
    """
    if ':' in col:
        name, idxs_str = col.split(':', 1)
        idxs = [slice(None)] + [int(idx) for idx in idxs_str.split(':')]
        return catalog[name][tuple(idxs)]
    else:
        return catalog[col]
