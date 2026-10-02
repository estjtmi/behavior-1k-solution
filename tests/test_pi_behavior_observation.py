import numpy as np

from b1k.models.observation import Observation
from b1k.policies.pi_behavior_policy import PiBehaviorPolicy
from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper, _state_for_correction_rules


def test_task_stage_prompt_is_batched_pair_for_model_indexing():
    wrapper = B1KPolicyWrapper(policy=object(), task_id=4)
    raw = wrapper.prepare_batch_for_pi_behavior({
        "observation/state": np.zeros((1, 32), dtype=np.float32),
        "observation/egocentric_camera": np.zeros((1, 224, 224, 3), dtype=np.uint8),
    })
    assert raw["tokenized_prompt"].shape == (2,)
    assert raw["tokenized_prompt"].tolist() == [4, 0]
    assert raw["tokenized_prompt_mask"].tolist() == [True, True]

    policy = object.__new__(PiBehaviorPolicy)
    policy._input_transform = lambda data: data
    prepared = policy._prepare_inputs(raw)
    assert prepared["tokenized_prompt"].shape == (1, 2)
    assert prepared["tokenized_prompt"].tolist() == [[4, 0]]
    assert prepared["tokenized_prompt_mask"].shape == (1, 2)
    assert prepared["tokenized_prompt_mask"].tolist() == [[True, True]]


def test_observation_accepts_batched_task_stage_pair():
    observation = Observation.from_dict({
        "image": {
            "base_0_rgb": np.zeros((1, 224, 224, 3), dtype=np.float32),
            "left_wrist_0_rgb": np.zeros((1, 224, 224, 3), dtype=np.float32),
            "right_wrist_0_rgb": np.zeros((1, 224, 224, 3), dtype=np.float32),
        },
        "image_mask": {
            "base_0_rgb": np.ones((1,), dtype=bool),
            "left_wrist_0_rgb": np.ones((1,), dtype=bool),
            "right_wrist_0_rgb": np.ones((1,), dtype=bool),
        },
        "state": np.zeros((1, 32), dtype=np.float32),
        "tokenized_prompt": np.array([[4, 0]], dtype=np.int32),
        "tokenized_prompt_mask": np.array([[True, True]], dtype=bool),
    })
    assert observation.tokenized_prompt.shape == (1, 2)


def test_correction_rules_receive_unbatched_32d_state():
    state = _state_for_correction_rules(np.zeros((1, 256), dtype=np.float32))
    assert state.shape == (23,)
