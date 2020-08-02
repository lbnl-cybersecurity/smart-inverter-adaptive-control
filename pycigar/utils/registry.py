from pycigar.utils.pycigar_registration import pycigar_register


def register_devcon(devcon_name, devcon_class, **kwargs):

    assert isinstance(devcon_name, str)

    try:
        pycigar_register(
            id=devcon_name,
            entry_point='{}:{}'.format(devcon_class.__module__, devcon_class.__name__),
            kwargs=kwargs
            )
    except Exception:
        pass

    return devcon_name
