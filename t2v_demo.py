import os
import pathlib
import random
import shlex
import subprocess

import gradio as gr
import torch
from huggingface_hub import snapshot_download

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline

model_dir = pathlib.Path('weights')
if not model_dir.exists():
    model_dir.mkdir()
    
    snapshot_download('damo-vilab/modelscope-damo-text-to-video-synthesis',
                        repo_type='model',
                        local_dir=model_dir)

DESCRIPTION = '# [ModelScope Text to Video Synthesis](https://modelscope.cn/models/damo/text-to-video-synthesis/summary)'
DESCRIPTION += '\n<p>For Colab usage, you can view <a href="https://colab.research.google.com/drive/1uW1ZqswkQ9Z9bp5Nbo5z59cAn7I0hE6R?usp=sharing" style="text-decoration: underline;" target="_blank">this webpage</a>.（the latest update on 2023.03.21）</p>'
DESCRIPTION += '\n<p>This model can only be used for non-commercial purposes. To learn more about the model, take a look at the <a href="https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis" style="text-decoration: underline;" target="_blank">model card</a>.</p>'
if (SPACE_ID := os.getenv('SPACE_ID')) is not None:
    DESCRIPTION += f'\n<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. <a href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>'

pipe = pipeline('text-to-video-synthesis', model_dir.as_posix())

def generate(prompt: str, seed: int) -> str:
    print('11111111111')
    if seed == -1:
        seed = random.randint(0, 1000000)
    torch.manual_seed(seed)
    return pipe({'text': prompt},output_video='output_1000_seed2.mp4')[OutputKeys.OUTPUT_VIDEO]

# generate('An astronaut riding a horse.',0)
generate('a jeep car is moving on the road',2)

# examples = [
#     ['An astronaut riding a horse.', 0],
#     ['A panda eating bamboo on a rock.', 0],
#     ['Spiderman is surfing.', 0],
# ]

# with gr.Blocks(css='style.css') as demo:
#     gr.Markdown(DESCRIPTION)
#     with gr.Group():
#         with gr.Box():
#             with gr.Row(elem_id='prompt-container').style(equal_height=True):
#                 prompt = gr.Text(
#                     label='Prompt',
#                     show_label=False,
#                     max_lines=1,
#                     placeholder='Enter your prompt',
#                     elem_id='prompt-text-input').style(container=False)
#                 run_button = gr.Button('Generate video').style(
#                     full_width=False)
#         result = gr.Video(label='Result', show_label=False, elem_id='gallery')
#         with gr.Accordion('Advanced options', open=False):
#             seed = gr.Slider(
#                 label='Seed',
#                 minimum=-1,
#                 maximum=1000000,
#                 step=1,
#                 value=-1,
#                 info='If set to -1, a different seed will be used each time.')

#     inputs = [prompt, seed]
#     gr.Examples(examples=examples,
#                 inputs=inputs,
#                 outputs=result,
#                 fn=generate,
#                 cache_examples=os.getenv('SYSTEM') == 'spaces')

#     prompt.submit(fn=generate, inputs=inputs, outputs=result)
#     run_button.click(fn=generate, inputs=inputs, outputs=result)

#     with gr.Accordion(label='Biases and content acknowledgment', open=False):
#         gr.HTML("""<div class="acknowledgments">
#                     <h4>Biases and content acknowledgment</h4>
#                     <p>
#                         Despite how impressive being able to turn text into video is, beware to the fact that this model may output content that reinforces or exacerbates societal biases. The training data includes LAION5B, ImageNet, Webvid and other public datasets. The model was not trained to realistically represent people or events, so using it to generate such content is beyond the model's capabilities.
#                     </p>
#                     <p>
#                         It is not intended to generate content that is demeaning or harmful to people or their environment, culture, religion, etc. Similarly, it is not allowed to generate pornographic, violent and bloody content generation. <b>The model is meant for research purposes</b>.
#                     </p>
#                     <p>
#                         To learn more about the model, head to its <a href="https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis" style="text-decoration: underline;" target="_blank">model card</a>.
#                     </p>
#                    </div>
#                 """)

# demo.queue(api_open=False, max_size=15).launch(share=True,server_port=8080)