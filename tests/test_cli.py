import pytest
from compilerpilot.cli import main, prepare_data


def test_agent_credentials_checked_before_build(monkeypatch, tmp_path):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    assert main(['experiment','--strategy','agent','--model','fixed-test-model',
                 '--results',str(tmp_path)])==1
    assert not list(tmp_path.iterdir())


def test_training_seed_cannot_equal_evaluation_seed(tmp_path):
    with pytest.raises(ValueError,match='seeds must differ'):
        prepare_data(tmp_path,tmp_path/'data',100,1,1)


def test_command_limit_checked_before_build(tmp_path):
    with pytest.raises(ValueError,match='commands'):
        prepare_data(tmp_path,tmp_path/'data',0,1,2)
