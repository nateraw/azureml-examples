# import json
# import pytest
# from pathlib import Path

# from lightning_mnist_example import score, train


# @pytest.mark.usefixtures('rm_logdir')
# def test_local_run():
#     args = train.parse_args('--max_epochs 1 --default_root_dir logs/'.split())
#     train.main(args)
#     assert 1 == 1

#     ckpt_paths = list(Path('logs/lightning_logs/').glob('**/*.ckpt'))
#     assert bool(ckpt_paths) == True
#     assert len(ckpt_paths) == 1

#     score.init()
#     request_data = score.get_example_data()
#     response_data = score.run(request_data)
#     assert response_data['predictions'] == [6, 7]
