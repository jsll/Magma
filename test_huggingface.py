from PIL import Image
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor 
from scripts.utils import average_attention_heads,attention_rollout,combine_attention_matrices
import argparse


def test(head_fusion, random_text_prompt, random_input_image):
    action_norm_stats = {
        "bridge_orig": {'mask': [True, True, True, True, True, True, False], 'max': [0.41691166162490845, 0.25864794850349426, 0.21218234300613403, 3.122201919555664, 1.8618112802505493, 6.280478477478027, 1.0], 'mean': [0.0002334194869035855, 0.00013004911306779832, -0.00012762474943883717, -0.0001556558854645118, -0.0004039328487124294, 0.00023557482927571982, 0.5764579176902771], 'min': [-0.4007510244846344, -0.13874775171279907, -0.22553899884223938, -3.2010786533355713, -1.8618112802505493, -6.279075622558594, 0.0], 'q01': [-0.02872725307941437, -0.04170349963009357, -0.026093858778476715, -0.08092105075716972, -0.09288699507713317, -0.20718276381492615, 0.0], 'q99': [0.028309678435325586, 0.040855254605412394, 0.040161586627364146, 0.08192047759890528, 0.07792850524187081, 0.20382574498653397, 1.0], 'std': [0.009765930473804474, 0.013689135201275349, 0.012667362578213215, 0.028534092009067535, 0.030637972056865692, 0.07691419124603271, 0.4973701536655426]},
        "google_robot": {'mask': [True, True, True, True, True, True, False], 'max': [2.9984593391418457, 22.09052848815918, 2.7507524490356445, 1.570636510848999, 1.5321086645126343, 1.5691522359848022, 1.0], 'mean': [0.006987582892179489, 0.006265917327255011, -0.01262515690177679, 0.04333311319351196, -0.005756212864071131, 0.0009130256366916001, 0.5354204773902893], 'min': [-2.0204520225524902, -5.497899532318115, -2.031663417816162, -1.569917917251587, -1.569892168045044, -1.570419430732727, 0.0], 'q01': [-0.22453527510166169, -0.14820013284683228, -0.231589707583189, -0.3517994859814644, -0.4193011274933815, -0.43643461108207704, 0.0], 'q99': [0.17824687153100965, 0.14938379630446405, 0.21842354819178575, 0.5892666035890578, 0.35272657424211445, 0.44796681255102094, 1.0], 'std': [0.0692116990685463, 0.05970962345600128, 0.07353084534406662, 0.15610496699810028, 0.13164450228214264, 0.14593800902366638, 0.497110515832901]}
    }


    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained("microsoft/Magma-8B", trust_remote_code=True, torch_dtype=dtype)
    processor = AutoProcessor.from_pretrained("microsoft/Magma-8B", trust_remote_code=True)
    model.to("cuda")
    vocab_size = processor.tokenizer.vocab_size

    # Inference
    image = Image.open("./assets/images/magma_logo.jpg").convert("RGB")
    image = image.resize((256, 256))
    image_size = 256
    patch_size = 16 # This may vary based on the model's vision encoder
    num_patches_h = image_size // patch_size
    num_patches_w = image_size // patch_size
    total_patches = num_patches_h * num_patches_w

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
        action_ids = outputs.sequences[0, -8:-1].cpu().tolist()
    
        attention_matrices = outputs.attentions  # This will contain the attention matrices
        combined_attention_matrices = combine_attention_matrices(attention_matrices )
        layer_wise_average_attention = average_attention_heads(combined_attention_matrices, head_fusion=head_fusion)
        attention_rollout_matrix = attention_rollout(layer_wise_average_attention)

        action_to_image_attention = attention_rollout_matrix[0,-8:-1, patches_start:patches_end]
        action_to_text_attention = attention_rollout_matrix[0,-8:-1, patches_start:patches_end]
        image_patches_attention = attention_rollout_matrix[:, patches_start:patches_end]
        patch_attention = image_patches_attention.reshape(-1, num_patches_h, num_patches_w)
        breakpoint()
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
