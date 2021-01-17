import inspect

import gin
import six


# TODO: Delete this file when https://github.com/google/gin-config/pull/48 gets merged
def _decorate_fn_or_cls(decorator, fn_or_cls, subclass=False):
    """Decorate a function or class with the given decorator.

    When `fn_or_cls` is a function, applies `decorator` to the function and
    returns the (decorated) result.

    When `fn_or_cls` is a class and the `subclass` parameter is `False`, this will
    replace `fn_or_cls.__init__` with the result of applying `decorator` to it.

    When `fn_or_cls` is a class and `subclass` is `True`, this will subclass the
    class, but with `__init__` defined to be the result of applying `decorator` to
    `fn_or_cls.__init__`. The decorated class has metadata (docstring, name, and
    module information) copied over from `fn_or_cls`. The goal is to provide a
    decorated class the behaves as much like the original as possible, without
    modifying it (for example, inspection operations using `isinstance` or
    `issubclass` should behave the same way as on the original class).

    Args:
      decorator: The decorator to use.
      fn_or_cls: The function or class to decorate.
      subclass: Whether to decorate classes by subclassing. This argument is
        ignored if `fn_or_cls` is not a class.

    Returns:
      The decorated function or class.
    """
    if not inspect.isclass(fn_or_cls):  # pytype: disable=wrong-arg-types
        return decorator(gin.config._ensure_wrappability(fn_or_cls))

    construction_fn = gin.config._find_class_construction_fn(fn_or_cls)

    if subclass:
        klass = fn_or_cls
        while hasattr(klass, "_gin_decorated") and klass._gin_decorated:
            klass = klass.__bases__[0]

        class DecoratedClass(fn_or_cls):
            __doc__ = fn_or_cls.__doc__
            __module__ = fn_or_cls.__module__
            __class__ = klass

            def __reduce__(self):
                return super(DecoratedClass, self).__reduce__()

        DecoratedClass.__name__ = fn_or_cls.__name__
        DecoratedClass._gin_decorated = True
        DecoratedClass.__qualname__ = fn_or_cls.__qualname__
        cls = DecoratedClass
    else:
        cls = fn_or_cls

    decorated_fn = decorator(gin.config._ensure_wrappability(construction_fn))
    if construction_fn.__name__ == "__new__":
        decorated_fn = staticmethod(decorated_fn)
    setattr(cls, construction_fn.__name__, decorated_fn)
    return cls


gin.config._decorate_fn_or_cls = _decorate_fn_or_cls
