import os
import time

import pytest
from text_generation_server.generator import TpuGenerator
from text_generation_server.pb.generate_pb2 import (
    Batch,
    NextTokenChooserParameters,
    Request,
    StoppingCriteriaParameters,
)
from tqdm import tqdm

from optimum.tpu.model import fetch_model


MODEL_ID = "google/gemma-7b"
SEQUENCE_LENGTH = 128


@pytest.fixture(scope="module")
def model_path():
    # Add variables to environment so they can be used in AutoModelForCausalLM
    os.environ["HF_SEQUENCE_LENGTH"] = str(SEQUENCE_LENGTH)
    path = fetch_model(MODEL_ID)
    return path


def create_request(
    id: int,
    inputs: str,
    max_new_tokens=20,
    do_sample: bool = True, # False,
    top_k: int = 1, # 50,
    top_p: float = 1, # 0.9,
    temperature: float = 1.0,
    seed: int = 0,
    repetition_penalty: float = 1.0,
):
    # For these tests we can safely set typical_p to 1.0 (default)
    typical_p = 1.0
    if not do_sample:
        # Drop top_p parameter to avoid warnings
        top_p = 1.0
    parameters = NextTokenChooserParameters(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        seed=seed,
        repetition_penalty=repetition_penalty,
        typical_p=typical_p,
    )
    stopping_parameters = StoppingCriteriaParameters(max_new_tokens=max_new_tokens)
    return Request(id=id, inputs=inputs, parameters=parameters, stopping_parameters=stopping_parameters)

def do_simulation(prompts, replys, prefill_bucket_size_to_ms, system_time_per_decode_token_ms):
  # import pdb; pdb.set_trace()
  def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

  def tokens_in_input_str(s):
    import pdb; pdb.set_trace()
    return_val = int(1.3 * len(s.split()))
    return return_val

  convo_numbers = []
  # Please update with your own data file path

  # with open(sharegpt_path, "r", encoding="utf-8") as f:
  #   loaded_share_gpt = json.load(f)
  # for example in prompts:
  for i in range(len(prompts)):
    # if len(example["conversations"]) < 2:
    #   continue
    input_tokens = tokens_in_input_str(prompts[i])
    output_tokens = tokens_in_input_str(replys[i].text)
    convo_numbers.append((input_tokens, output_tokens))

  num_convos = len(convo_numbers)
  kept_convos = [
      c for c in convo_numbers # if c[0] <= CUTOFF_INPUT and c[1] <= CUTOFF_OUTPUT # CUTOFF_INPUT = 1024 # CUTOFF_OUTPUT = 1024
  ]

  mean_input = sum(c[0] for c in kept_convos) / len(kept_convos)
  mean_output = sum(c[1] for c in kept_convos) / len(kept_convos)

  print(
      f"""Total {num_convos=} but only kept {kept_convos=}. 
    Out of kept, {mean_input=}, {mean_output=}"""
  )

  total_prefill_system_ms = 0
  total_generate_system_ms = 0

  for convo in kept_convos:
    input_tok, output_tok = convo
    bucket = max(128, next_power_of_2(input_tok))
    generate_system_ms = output_tok * system_time_per_decode_token_ms
    prefill_system_ms = prefill_bucket_size_to_ms[bucket]

    print(
        f"{convo=} {bucket=}, {prefill_system_ms=:.2f}, {generate_system_ms=:.2f}"
    )

    total_prefill_system_ms += prefill_system_ms
    total_generate_system_ms += generate_system_ms

  total_time_ms = total_prefill_system_ms + total_generate_system_ms
  input_tokens = sum(c[0] for c in kept_convos)

  output_tokens = sum(c[1] for c in kept_convos)
  print(
      f"""Output tokens {output_tokens} in {total_time_ms/1000:.2f} seconds, 
      for {output_tokens/(total_time_ms/1000):.2f} out tok/s"""
  )

  total_prefill_sec = total_prefill_system_ms / 1000
  total_generate_sec = total_generate_system_ms / 1000

  print(
      f"""Total time {total_time_ms/1000:.2f} seconds, 
      split {total_prefill_sec=:.2f} seconds and {total_generate_sec=:.2f} seconds"""
  )

  idealized_prefill_sec = (
      1.1 * input_tokens / 1024 * prefill_bucket_size_to_ms[1024] / 1000
  )

  prefill_savings_sec = total_prefill_sec - idealized_prefill_sec

  idealized_generate_sec = (
      total_generate_sec / 2
  )  # (Roughly save 75% on KV cache high cost on the rest)
  generate_savings_sec = total_generate_sec - idealized_generate_sec

  print(
      f"""we think prefill will take {total_prefill_sec=:.2f}, 
    we could get it to {idealized_prefill_sec=:.2f} so we'd 
    save {prefill_savings_sec=:.2f} seconds """
  )
  print(
      f"""with sparsity we could go from  {total_generate_sec=:.2f}, 
    we could get it to {idealized_generate_sec=:.2f} so we'd save 
    {generate_savings_sec=:.2f} seconds """
  )

  idealized_overall_time = idealized_generate_sec + idealized_prefill_sec

  print(
      f"""Idealized out tokens {output_tokens} in {idealized_overall_time:.2f} seconds, 
    for {output_tokens/idealized_overall_time:.2f} out tok/s"""
  )
  print("prfill", prefill_bucket_size_to_ms)
  print("decode step", system_time_per_decode_token_ms)

def test_run_decode_multi_all():
  os.environ["HF_SEQUENCE_LENGTH"] = str(SEQUENCE_LENGTH)
  model_path = fetch_model(MODEL_ID)
  # print("in test_decode_multi, model_path is: ", model_path)
  run_decode_multi(model_path)

# @pytest.mark.slow
def run_decode_multi(model_path):
    generator = TpuGenerator.from_pretrained(
        model_path, revision="", max_batch_size=1,
        max_sequence_length=SEQUENCE_LENGTH
    )

    start_time = time.time()
    prompts: List[str] = [
      "I believe the meaning of life is",
      "To add an element to an ArrayList of a specific class type in Java",
      "you can follow the following steps:.",
      "Create an instance of the class to be added.\n2.",
      "Get a reference to the ArrayList.\n3.", 
      "Call the `add()` method on the ArrayList,",
      "passing the instance of the class as the argument.",
      "Here's an example of how to add an object of type `Person` to an ArrayList of type `ArrayList<Person>`:",
      "```csharp\n// Create a new instance of the Person class\nPerson person = new Person(\"John\", 25);",
      "// Get a reference to the ArrayList\nArrayList<Person> peopleList = new ArrayList<>();",
      "Add the person object to the ArrayList peopleList",
      "In this example, the `Person` class is assumed to have a constructor that takes two arguments:",
      "a String for the person's name, and an int for their age. You can substitute your own class and constructor as necessary.",
      "You are an AI assistant. User will you give you a task.",
      "Your goal is to complete the task as faithfully as you can.",
      "While performing the task think step-by-step and justify your steps.\n<</SYS>>",
      "Question 1: What is commercial real estate finance?",
      "Question 2: What are Commercial Real Estate services?",
      "Options are:\n[a]. no.\n[b]. yes.\nWould the answer to these two questions be the same?",
      "You are an AI assistant that helps people find information.",
      "Provide a detailed answer so user don\u2019t need to search outside to understand the answer.",
      "Use reasoning to lead to the answer of the following question:\nWhere are you likely to find water underneath?",
      "Options:\n- toilet\n- sink\n- jar\n- bridge\n- house\n Reasoning process:",
      "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.",
      "Continue the following story.\n\nKay didn't have shoes that fit her feet properly.",
      "She only wore sneakers, because the \nChoose from: [I] shoes  fitted badly. [II] sneakers  fitted badly",
    ]
    # input_text = "It was a bright cold day in April, and the clocks were striking thirteen."
    max_new_tokens = 20
    # generated_text = "\n\nThe time is 1984. The place is Airstrip One, the British"

    # generator = TpuGenerator.from_pretrained(
    #     model_path, revision="", max_batch_size=1,
    #     max_sequence_length=SEQUENCE_LENGTH
    # )
    prefill_times = {}
    dec_times = []
    replys = []
    for prompt in prompts:
        input_text = prompt
        request = create_request(id=0, inputs=input_text, max_new_tokens=max_new_tokens, do_sample=True)
        batch = Batch(id=0, requests=[request], size=1, max_tokens=SEQUENCE_LENGTH)

        start_time_prefill = time.time()
        generations, next_batch = generator.prefill(batch)
        prefill_time = time.time() - start_time_prefill
        print("--- prefill time : %s seconds ---" % prefill_time, " for prompt: ", prompt)
        prefill_times[prompt] = prefill_time

        # We already generated one token: call decode max_new_tokens - 1 times
        start_time_decode = time.time()
        for _ in tqdm(range(max_new_tokens - 1)):
            assert next_batch.size == 1
            assert next_batch.max_tokens == SEQUENCE_LENGTH
            assert len(generations) == 1
            assert len(generations[0].tokens.ids) == 1
            generations, next_batch = generator.decode([next_batch])
        assert next_batch is None
        assert len(generations) == 1
        output = generations[0].generated_text
        replys.append(output)
        generator.clear()
        decode_time = time.time() - start_time_decode
        dec_times.append(decode_time)
        print("--- finish all decode used : %s seconds ---" % decode_time)
    print("--- finish all requests used : %s seconds ---" % (time.time() - start_time))
        # assert output.generated_tokens == max_new_tokens
        # assert output.finish_reason == 0
        # assert output.text == generated_text
    print("decode", sum(dec_times) / 10)

    prefill_times_ms = {k: v * 1000 for k, v in prefill_times.items()}
    decode_time_ms = sum(dec_times) * 1000 / 10 / 1 # FLAGS.batch_size

    # call fun
    do_simulation(prompts, replys, prefill_times_ms, decode_time_ms)

def main():
  os.environ["HF_SEQUENCE_LENGTH"] = str(SEQUENCE_LENGTH)
  start_time_load_model = time.time()
  model_path = fetch_model(MODEL_ID)
  print("--- load model used : %s seconds ---" % (time.time() - start_time_load_model))
  run_decode_multi(model_path)

if __name__ == '__main__':
  main()
