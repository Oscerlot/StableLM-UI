from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

_modelFolder = "models/"

#is force-set to base-3b currently just for the initial push (should run on 8GB VRAM)
modelname = _modelFolder + "stablelm-base-alpha-3b"
model = AutoModelForCausalLM.from_pretrained(modelname)
tokenizer = AutoTokenizer.from_pretrained(modelname)
model.half().cuda()

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    tokens = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True, )

    return tokenizer.decode(tokens[0], skip_special_tokens=True)

#add in options to change the temp/token count/model select, etc.
stableLM_UI = gr.Interface(
    fn=generate,
    inputs="text",
    outputs="text")

stableLM_UI.launch()
