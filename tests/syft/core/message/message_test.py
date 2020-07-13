from syft.core.message import (
    DeleteObjectMessage,
    GetObjectMessage,
    RunClassMethodMessage,
    RunFunctionOrConstructorMessage,
    SaveObjectMessage,
)


def test_delete_message() -> None:
    message = DeleteObjectMessage(id="1")
    assert message.id == "1"


def test_get_message() -> None:
    message = GetObjectMessage(id="2")
    assert message.id == "2"


def test_run_class() -> None:
    any_object = set([1, 2, 3])
    message = RunClassMethodMessage(
        path="test_path",
        _self=any_object,
        args=["a", "b"],
        kwargs={"arg1": "val1", "arg2": "val2"},
        id_at_location="3",
    )

    assert message.path == "test_path"
    assert message._self == set([1, 2, 3])
    assert type(message._self) == set
    assert message.args == ["a", "b"]
    assert message.kwargs == {"arg1": "val1", "arg2": "val2"}
    assert message.id_remote == "3"


def test_run_function() -> None:
    message = RunFunctionOrConstructorMessage(
        path="test_path", args=["a", "b"], kwargs={"arg1": "val1", "arg2": "val2"},
    )

    assert message.path == "test_path"
    assert message.args == ["a", "b"]
    assert message.kwargs == {"arg1": "val1", "arg2": "val2"}


def test_save_message() -> None:
    any_object = set([4, 5, 6])
    message = SaveObjectMessage(id="4", obj=any_object)

    assert message.id == "4"
    assert message.obj == set([4, 5, 6])
    assert type(message.obj) == set
