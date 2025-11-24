import sys, json
from riichi_discard_wisdom.cli_defense import main

def test_defense_top_only_json(monkeypatch, capsys):
    argv = ["prog", "--groups", "m123456789p11s11", "--json", "--top-only"]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "best" in data and data["best"]["label"]
