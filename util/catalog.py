def get_column(catalog, *cols):
    """
    Convenience function to allow accessing multidimensional cells
    """
    r = []
    for col in cols:
        if ':' in col:
            name, idxs_str = col.split(':', 1)
            idxs = [slice(None)] + [int(idx) for idx in idxs_str.split(':')]
            r.append(catalog[name][tuple(idxs)])
        else:
            r.append(catalog[col])
    if len(r) == 1:
        return r[0]
    return r
