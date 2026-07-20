from devmlops_ci.calculator import add, subtract


def test_add() -> None:
    assert add(2, 3) == 5


def test_add_negative() -> None:
    assert add(-1, 1) == 0


def test_subtract() -> None:
    assert subtract(5, 3) == 2
