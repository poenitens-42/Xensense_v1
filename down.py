from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "microsoft/phi-4-mini-instruct"

print("[INFO] Loading Phi-4-Mini model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="cpu",             # << forces CPU loading
)
print("[INFO] Model loaded successfully.")
