from lungscan3d.commands import Commands, _load_config, _with_default_data_luna16


def test_fire_commands_expose_config_only_methods():
    commands = Commands()

    assert callable(commands.train)
    assert callable(commands.export_tensorrt)
    assert callable(commands.self_test)


def test_download_luna16_defaults_to_luna16_config():
    assert _with_default_data_luna16(["data.download_max_subsets=1"]) == [
        "data=luna16",
        "data.download_max_subsets=1",
    ]


def test_default_logging_is_disabled():
    config = _load_config([])

    assert config.logging.mode == "none"
    assert config.logging.mlflow_port == 8080
    assert config.logging.tensorboard_port == 6006


def test_luna16_defaults_fit_midrange_workstation():
    config = _load_config(["data=luna16"])

    assert config.data.batch_size == 64
    assert config.preprocessing.chunk_size == 256
    assert config.data.split_by_patient is True
