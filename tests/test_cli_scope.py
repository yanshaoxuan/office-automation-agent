"""CLI --scope 参数解析测试。

背景：Windows PowerShell 5.1 向原生 exe 传参会吞内层双引号，
JSON 写法在 PS 下不可靠，因此 _parse_scope 支持 shell 友好的
key=value 格式，同时保留 JSON 兼容。
"""
from __future__ import annotations

from office_agent.main import _parse_scope


def test_key_value_basic():
    assert _parse_scope("department=IT") == {"department": "IT"}


def test_key_value_multiple():
    assert _parse_scope("department=IT,doc_type=faq") == {
        "department": "IT",
        "doc_type": "faq",
    }


def test_key_value_with_spaces():
    assert _parse_scope(" department = IT ") == {"department": "IT"}


def test_json_still_supported():
    """JSON 格式保留兼容（bash / 已正确转义的场景）。"""
    assert _parse_scope('{"department":"IT"}') == {"department": "IT"}


def test_powershell_mangled_json_is_rejected():
    """PS 吞引号后的残骸 {department:IT} 应明确失败，而不是瞎猜。"""
    assert _parse_scope("{department:IT}") is None


def test_invalid_inputs_rejected():
    assert _parse_scope("department") is None
    assert _parse_scope("department=") is None
    assert _parse_scope("=IT") is None
    assert _parse_scope("") is None
    assert _parse_scope("   ") is None
