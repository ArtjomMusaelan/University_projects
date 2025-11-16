"""
Focused on checking system_prompts (50 example of each emotion, llama3.2:3b, English text):
=== Results for system_prompt ===
F1 Score: 0.2963
=== Results for system_few_shot_prompt ===
F1 Score: 0.4389

Added control of generation parameters: temperature: 0.0, top_p: 0.9, top_k: 50, repetition_penalty: 1.1:
=== Results for system_prompt ===
F1 Score: 0.3159
=== Results for system_few_shot_prompt ===
F1 Score: 0.4031

Changed repetition_penalty: 1.0:
=== Results for system_prompt ===
F1 Score: 0.3412
=== Results for system_few_shot_prompt ===
F1 Score: 0.4083

Changed repetition_penalty: 1.0, top_k: 400:
=== Results for system_prompt ===
F1 Score: 0.3059
=== Results for system_few_shot_prompt ===
F1 Score: 0.4225

Changed 10 examples of each emotion:
=== Results for system_prompt ===
F1 Score: 0.3364
=== Results for system_few_shot_prompt ===
F1 Score: 0.4327

Added “context_length”: 100:
=== Results for system_prompt ===
F1 Score: 0.3055
=== Results for system_few_shot_prompt ===
F1 Score: 0.4563

Try “context_length”: 2048:
=== Results for system_prompt ===
F1 Score: 0.2639
=== Results for system_few_shot_prompt ===
F1 Score: 0.4399

Try “context_length”: 1000:
=== Results for system_prompt ===
F1 Score: 0.3256
=== Results for system_few_shot_prompt ===
F1 Score: 0.4634

Trying “context_length”: 1500:
=== Results for system_prompt ===
F1 Score: 0.3230
=== Results for system_few_shot_prompt ===
F1 Score: 0.3959

Trying “context_length”: 500:
=== Results for system_prompt ===
F1 Score: 0.2752
=== Results for system_few_shot_prompt ===
F1 Score: 0.4227

Trying “context_length”: 750:
=== Results for system_prompt ===
F1 Score: 0.3038
=== Results for system_few_shot_prompt ===
F1 Score: 0.4070

Trying “context_length”: 1250:
=== Results for system_prompt ===
F1 Score: 0.2690
=== Results for system_few_shot_prompt ===
F1 Score: 0.4600

Trying “context_length”: 1125:
=== Results for system_prompt ===
F1 Score: 0.2954
=== Results for system_few_shot_prompt ===
F1 Score: 0.4133

Best result with context_length 1000 - I leave it
"""