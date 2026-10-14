from src.parser import parse_plate


def test_basic():
    r = parse_plate("22-A 73456")
    assert r["code"] == "22" and r["letter"] == "A" and r["number"] == "73456"


def test_arabic_digits():
    assert parse_plate("٢٢ A ٧٣٤٥٦")["formatted"] == "22-A 73456"


def test_invalid():
    assert parse_plate("hello") is None
