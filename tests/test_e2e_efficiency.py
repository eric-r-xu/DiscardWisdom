import sys, json
from riichi_discard_wisdom.cli_efficiency import main

def test_efficiency_top_only(monkeypatch, capsys):
    argv = ["prog", "--groups", "m123m456m789p11s11", "--json"]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "best" in data and data["best"]["label"]
