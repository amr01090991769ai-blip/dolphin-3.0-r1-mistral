import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "dphn/Dolphin3.0-R1-Mistral-24B"

# تحميل التوكنايزر
tokenizer = AutoTokenizer.from_pretrained(model_id)

# تحميل الموديل (يتطلب ذاكرة فيديو كبيرة أو استخدام 4-bit quantization)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# تجهيز الرسالة بتنسيق ChatML
messages = [
    {"role": "system", "content": "You are Dolphin, a helpful AI assistant."},
    {"role": "user", "content": "Explain the concept of quantum entanglement in simple terms."}
]

# تحويل الرسائل إلى تنسيق الموديل
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# إنشاء الإجابة
outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    temperature=0.1,
    top_p=0.9,
    do_sample=True
)

response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
print(response)
