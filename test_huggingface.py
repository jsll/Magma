from PIL import Image
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor 
from scripts.utils import average_attention_heads,attention_rollout,combine_attention_matrices
import argparse
import datasets

def test(head_fusion, random_text_prompt, random_input_image):

    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained("microsoft/Magma-8B", trust_remote_code=True, torch_dtype=dtype)
    processor = AutoProcessor.from_pretrained("microsoft/Magma-8B", trust_remote_code=True, padding=False)
    model.to("cuda")
    vocab_size = processor.tokenizer.vocab_size

    convs = [
        {"role": "system", "content": "You are an agent that can see, talk and act."},            
        {"role": "user"},
    ]
    generation_args = { 
        "max_new_tokens": 500, 
        "temperature": 0.0, 
        "do_sample": False, 
        "use_cache": True,
        "num_beams": 1,
        "output_attentions": True,  # Add this line to get attention matrices
        "output_hidden_states":True,
    } 

    if not random_input_image:
        dataset_names = [
            "fractal20220817_data",
            "stanford_hydra_dataset_converted_externally_to_rlds",
        ]
        data_per_dataset = {}
        for dataset_name in dataset_names:
            ds = datasets.load_dataset(
                "jxu124/OpenX-Embodiment",
                dataset_name,
                streaming=True,
                split="train",
            )  # IterDataset
            random_item = next(itertools.islice(ds, 10, 10 + 50))
            data_per_dataset[dataset_name] = random_item
        # Grab image input & format prompt
        # data = data_per_dataset["stanford_hydra_dataset_converted_externally_to_rlds"]
        data = data_per_dataset["fractal20220817_data"]
        image = Image.open(io.BytesIO(data["data.pickle"]["steps"][-1]["observation"]["image"]["bytes"]))
        image = image.resize((256, 256))

        # Inference
        #image = Image.open("./assets/images/magma_logo.jpg").convert("RGB")
        #image = image.resize((256, 256))
        #image_size = 256
        #patch_size = 16 # This may vary based on the model's vision encoder
        #num_patches_h = image_size // patch_size
        #num_patches_w = image_size // patch_size
        #total_patches = num_patches_h * num_patches_w

    while True:
        if not random_text_prompt:
            object_prompt = input("What object should the robot affect? ")
            action_prompt = input("What action should the robot take? ")
            if action_prompt == "q":
                break
            prompt = f"In: What action should the robot take to {action_prompt} the {object_prompt}?\nOut:"
            # calibration_prompt = "What action should the robot take to <unk> the <unk>?\nOut:"
            calibration_prompt = "In: What action should the robot take to '' the ''?\nOut:"
        else:
            prompt = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
            action_prompt = prompt
        if random_input_image:
            random_array = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(random_array)

        convs[1]["content"] = f"<image>\nWhat action should the robot take to {action_prompt} the {object_prompt}?"
        prompt = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=[image], texts=prompt, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
        inputs = inputs.to("cuda").to(dtype)

        decoded_input_tokens = [processor.tokenizer.decode([token_id], skip_special_tokens=False) for token_id in inputs.input_ids[0]       ]

        input_sequence = inputs["input_ids"][0]
        image_token_position = torch.where(input_sequence == model.config.image_token_index)[0].item()
        patches_start = image_token_position
        patches_end = patches_start + total_patches

        with torch.inference_mode():
            outputs= model.generate(**inputs, **generation_args, return_dict_in_generate=True)


        decoded_output_tokens = [processor.tokenizer.decode([token_id], skip_special_tokens=False) for token_id in outputs.sequences[0]]
        #action_ids = outputs.sequences[0, -8:-1].cpu().tolist()
    
        attention_matrices = outputs.attentions  # This will contain the attention matrices
        combined_attention_matrices = combine_attention_matrices(attention_matrices )
        layer_wise_average_attention = average_attention_heads(combined_attention_matrices, head_fusion=head_fusion)
        attention_rollout_matrix = attention_rollout(layer_wise_average_attention)

        attention_rollout_from_actions_to_all_inputs_except_actions = attention_rollout_matrix[0,-8:-1,:-8]
        average_attention_from_actions_to_all_inputs_except_actions = attention_rollout_from_actions_to_all_inputs.mean(dim=0)
        # Entropy of the attention from actions to all inputs except actions
        entropy = -torch.sum(average_attention_from_actions_to_all_inputs_except_actions * torch.log(average_attention_from_actions_to_all_inputs_except_actions))
        print("Entropy:", entropy)
        print("Std:", average_attention_from_actions_to_all_inputs_except_actions.std())
        #action_to_image_attention = attention_rollout_matrix[0,-8:-1, patches_start:patches_end]
        #action_to_text_attention = attention_rollout_matrix[0,-8:-1, patches_start:patches_end]
        #image_patches_attention = attention_rollout_matrix[:, patches_start:patches_end]
        #patch_attention = image_patches_attention.reshape(-1, num_patches_h, num_patches_w)
        #breakpoint()
        #print("Attention shape:", [att for att in attention_matrices])  # Print the shapes of attention matrices

if __name__ == "__main__":
    # Make an argparser that takes as input the possible prompt to openvla
    # and the image to be used
    args = None
    parser = argparse.ArgumentParser()
    parser.add_argument("--head_fusion", type=str, choices=["mean", "max", "min"], default="mean")
    parser.add_argument("--set_random_text", action="store_true")
    parser.add_argument("--set_random_image", action="store_true")
    args = parser.parse_args()
    test(args.head_fusion, args.set_random_text, args.set_random_image)
