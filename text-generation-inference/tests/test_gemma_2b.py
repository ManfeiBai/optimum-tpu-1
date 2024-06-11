import os
# import pytest
import time
from text_generation_server.generator import TpuGenerator
from text_generation_server.pb.generate_pb2 import (
    Batch,
    NextTokenChooserParameters,
    Request,
    StoppingCriteriaParameters,
)
from tqdm import tqdm
from optimum.tpu.model import fetch_model

MODEL_ID = "google/gemma-2b"
SEQUENCE_LENGTH = 1024

def model_path():
    # Add variables to environment so they can be used in AutoModelForCausalLM
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
    # print("just before request")
    return Request(id=id, inputs=inputs, parameters=parameters, stopping_parameters=stopping_parameters)

def test_run_decode_multi_all():
  os.environ["HF_SEQUENCE_LENGTH"] = str(SEQUENCE_LENGTH)
  model_path = fetch_model(MODEL_ID)
  # print("in test_decode_multi, model_path is: ", model_path)
  run_decode_multi(model_path)

def run_decode_multi(model_path):
  os.environ["HF_SEQUENCE_LENGTH"] = str(SEQUENCE_LENGTH)
  model_path = fetch_model(MODEL_ID)
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
  generator = TpuGenerator.from_pretrained(model_path, revision="", max_batch_size=1, max_sequence_length=SEQUENCE_LENGTH)
  for prompt in prompts:
    input_text = prompt
    max_new_tokens = 20

    request = create_request(id=0, inputs=input_text, max_new_tokens=max_new_tokens, do_sample=False)
    batch = Batch(id=0, requests=[request], size=1, max_tokens=SEQUENCE_LENGTH)
    generations, next_batch = generator.prefill(batch)
    for _ in tqdm(range(max_new_tokens - 1)):
        assert next_batch.size == 1
        assert next_batch.max_tokens == 1024
        assert len(generations) == 1
        assert len(generations[0].tokens.ids) == 1
        generations, next_batch = generator.decode([next_batch])
    assert next_batch is None
    assert len(generations) == 1
    output = generations[0].generated_text
    generator.clear()
  print("--- finish all requests used : %s seconds ---" % (time.time() - start_time))

def main():
  os.environ["HF_SEQUENCE_LENGTH"] = str(SEQUENCE_LENGTH)
  model_path = fetch_model(MODEL_ID)
  run_decode_multi(model_path)

if __name__ == '__main__':
  main()
