import spdivik.types as ty


def minimal_size(data: ty.Data, size: int=2) -> bool:
    return data.shape[0] <= size
