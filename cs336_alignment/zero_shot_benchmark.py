from vllm import LLM, SamplingParams
import pathlib
from cs336_alignment.utils import load_math_validation_set, save_jsonl
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

MODEL_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "models" / "Qwen2.5-Math-1.5"
OUPUT_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "outputs"
NUM_RUNS = 5
ds = load_math_validation_set()
sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True)
llm = LLM(model=str(MODEL_PATH))

def get_prompt(question):
  return f"""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
  User: {question}
  Assistant: <think>"""

# ds = ds[:32]
prompts = [get_prompt(example['problem']) for example in ds]
results = []
for run_id in range(NUM_RUNS):
   responses = llm.generate(prompts, sampling_params)
   for i in range(len(ds)):
      question = ds[i]["problem"]
      solution = ds[i]["solution"]
      model_response = responses[i].outputs[0].text
      r1_reward = r1_zero_reward_fn(model_response, solution)
      results.append(
         {
            "question_id": i,
            "run_id": run_id,
            "question": question,
            "solution": solution,
            "model_response": model_response,
            "r1_format_reward": r1_reward["format_reward"],
            "r1_answer_reward": r1_reward["answer_reward"],
            "r1_reward": r1_reward["reward"]
         }
      )
save_jsonl(results, OUPUT_PATH/"math_validation_zero_shot_test.jsonl")



