import pytest
import os
from text_generation_server.generator import TpuGenerator
from text_generation_server.model import fetch_model
from text_generation_server.pb.generate_pb2 import (
    Batch,
    NextTokenChooserParameters,
    Request,
    StoppingCriteriaParameters,
)


MODEL_ID = "google/gemma-2b"
BATCH_SIZE = 4
SEQUENCE_LENGTH = 1024


@pytest.fixture(scope="module")
def model_path():
    # Add variables to environment so they can be used in TpuModelForCausalLM
    os.environ["HF_BATCH_SIZE"] = str(BATCH_SIZE)
    os.environ["HF_SEQUENCE_LENGTH"] = str(SEQUENCE_LENGTH)
    path = fetch_model(MODEL_ID)
    return path


def create_request(
    id: int,
    inputs: str,
    max_new_tokens=20,
    do_sample: bool = False,
    top_k: int = 50,
    top_p: float = 0.9,
    temperature: float = 1.0,
    seed: int = 0,
    repetition_penalty: float = 1.0,
):
    parameters = NextTokenChooserParameters(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        seed=seed,
        repetition_penalty=repetition_penalty,
    )
    stopping_parameters = StoppingCriteriaParameters(max_new_tokens=max_new_tokens)
    return Request(id=id, inputs=inputs, parameters=parameters, stopping_parameters=stopping_parameters)


@pytest.mark.parametrize(
    "input_text, max_new_tokens, generated_text, do_sample",
    [
        [
            "It was a bright cold day in April, and the clocks were striking thirteen.",
            20,
            "\n\nThe sun was a man was a man was a man was a small town.\n\nThe first",
            False,
        ],
        [
            "It was a bright cold day in April, and the clocks were striking thirteen.",
            20,
            " We sat outside the house, drinking coffee, listening to the traffic. And then, suddenly, we",
            True,
        ],
    ],
    ids=["greedy", "sample"],
)
def test_decode_single(input_text, max_new_tokens, generated_text, do_sample, model_path):
    import time
    start = time.time()
    generator = TpuGenerator.from_pretrained(model_path)
    end = time.time()
    print(f"Model load took {end - start} seconds.")
    start = time.time()
    request = create_request(id=0, inputs=input_text, max_new_tokens=max_new_tokens, do_sample=do_sample)
    batch = Batch(id=0, requests=[request], size=1, max_tokens=SEQUENCE_LENGTH)
    end = time.time()
    print(f"batch setup took {end - start} seconds.")
    start = time.time()
    generations, next_batch = generator.prefill(batch)
    end = time.time()
    print(f"Prefill took {end - start} seconds.")
    start = time.time()
    # We already generated one token: call decode max_new_tokens - 1 times
    for i in range(max_new_tokens - 1):
        step_start = time.time()
        assert next_batch.size == 1
        assert next_batch.max_tokens == 1024
        assert len(generations) == 1
        assert len(generations[0].tokens.ids) == 1
        generations, next_batch = generator.decode([next_batch])
        step_end = time.time()
        print(f"Token {i} took {step_end - step_start} seconds.")
    end = time.time()
    print(f"total decode took {end - start} seconds.")
    print(f"Total number of tokens: {max_new_tokens - 1}")
    start = time.time()
    assert next_batch is None
    assert len(generations) == 1
    output = generations[0].generated_text
    print(f"Output text: {output.text}")
    assert output.generated_tokens == max_new_tokens
    assert output.finish_reason == 0
    # assert output.text == generated_text



def _test_decode_multiple(model_path):
    generator = TpuGenerator.from_pretrained(model_path)
    assert generator.model.config.batch_size > 1
    input_text = "Once upon a time"
    max_new_tokens = 20
    # Prefill a single request, remembering the generated token
    tokens = {0: [], 1: []}
    request = create_request(id=0, inputs=input_text, max_new_tokens=max_new_tokens)
    batch = Batch(id=0, requests=[request], size=1, max_tokens=SEQUENCE_LENGTH)
    generations, next_batch = generator.prefill(batch)
    assert next_batch.size == 1
    assert len(generations) == 1
    g = generations[0]
    tokens[g.request_id].append(g.tokens.ids[0])
    assert len(tokens[0]) == 1
    # Decode a few tokens
    gen_tokens = 4
    for _ in range(gen_tokens - 1):
        generations, next_batch = generator.decode([next_batch])
        assert len(generations) == 1
        g = generations[0]
        tokens[g.request_id].append(g.tokens.ids[0])
    assert len(tokens[0]) == gen_tokens
    assert next_batch.size == 1
    # Add a second request
    request = create_request(id=1, inputs=input_text, max_new_tokens=max_new_tokens)
    batch = Batch(id=1, requests=[request], size=1, max_tokens=SEQUENCE_LENGTH)
    generations, next_batch_1 = generator.prefill(batch)
    assert next_batch_1.size == 1
    # We should have generated only a single token
    assert len(generations) == 1
    g = generations[0]
    tokens[g.request_id].append(g.tokens.ids[0])
    assert len(tokens[0]) == gen_tokens
    assert len(tokens[1]) == 1
    # Decode more tokens until we reach the maximum for the first request
    batches = [next_batch, next_batch_1]
    for _ in range(max_new_tokens - gen_tokens):
        generations, next_batch = generator.decode(batches)
        for g in generations:
            tokens[g.request_id].append(g.tokens.ids[0])
        batches = [next_batch]
    # Verify we now only have one pending request
    assert next_batch.size == 1
    assert len(tokens[0]) == max_new_tokens
    assert len(tokens[1]) == max_new_tokens - gen_tokens + 1
    # Verify we have the output for the first request
    for g in generations:
        if g.request_id == 0:
            output = g.generated_text
            assert output.text != ""
            assert output.generated_tokens == max_new_tokens
            generated_text = output.text
    # Continue decoding until the end of the second request
    for _ in range(gen_tokens - 1):
        generations, next_batch = generator.decode([next_batch])
        assert len(generations) == 1
        g = generations[0]
        tokens[g.request_id].append(g.tokens.ids[0])
    assert next_batch is None
    output = generations[0].generated_text
    assert output.generated_tokens == max_new_tokens
    assert tokens[0] == tokens[1]
    assert output.text == generated_text