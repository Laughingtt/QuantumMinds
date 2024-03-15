from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM
import torch

model_dir = snapshot_download("Shanghai_AI_Laboratory/internlm2-chat-7b")