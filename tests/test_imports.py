import importlib


def test_import_snfs_traffic() -> None:
    import snfs_traffic

    assert snfs_traffic is not None


def test_version() -> None:
    import snfs_traffic

    assert hasattr(snfs_traffic, "__version__")
    assert snfs_traffic.__version__ == "0.1.0"


def test_subpackage_imports() -> None:
    modules = [
        "snfs_traffic.core",
        "snfs_traffic.topology",
        "snfs_traffic.scenarios",
        "snfs_traffic.rules",
        "snfs_traffic.observations",
        "snfs_traffic.envs",
        "snfs_traffic.metrics",
        "snfs_traffic.io",
    ]

    for module_name in modules:
        module = importlib.import_module(module_name)
        assert module is not None
