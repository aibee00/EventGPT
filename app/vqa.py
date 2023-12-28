"""
 # Copyright (c) 2022, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import streamlit as st
from app import load_demo_image, device
from app.utils import load_model_cache
from eventgpt.processors import load_processor
from PIL import Image

st.set_page_config(layout="wide")

SEP = "<bbox>"

def concat_question_bbox(question, bbox):
    if '{' in question:
        question = question.format(bbox)
        return f"{question}{SEP}{bbox}"
    return f"{question}{SEP}{bbox}"


def app():
    model_base = st.sidebar.selectbox("Model:", ["BLIP2", "BLIP"])
    image_size = st.sidebar.selectbox("Image size:", [224, 364, 480])
    max_len = st.sidebar.slider("Max length:", min_value=10, max_value=1024, value=512)  # 文本长度
    temperature = st.sidebar.slider("Temperature:", min_value=0.1, max_value=1.0, value=1.0)  # 温度

    model_name = st.sidebar.selectbox("Model name:", 
                                      ["blip2_opt_llava_box_caption_roi", 
                                       "blip2_opt_llava_box_caption",
                                       "blip_vqa", 
                                       "others"])  # 模型名称
    if model_name == "others":
        model_name = st.sidebar.text_input("Model name:", "blip2_opt_llava_box_caption")  # 模型名称

    model_type = st.sidebar.selectbox("Model type:", 
                                      ["pretrain_opt2.7b_llava_box_caption_roi", 
                                       "pretrain_opt2.7b_llava_box_caption",
                                       "pretrain_opt2.7b_llava_box_caption_vit_L364", 
                                       "vqav2", 
                                       "others"])  # 模型类型
    if model_type == "others":
        model_type = st.sidebar.text_input("Model type:", "pretrain_opt2.7b_llava_box_caption")  # 模型类型

    # ===== layout =====
    st.markdown(
        "<h1 style='text-align: center;'>Visual Question Answering</h1>",
        unsafe_allow_html=True,
    )

    instructions = """Try the provided image or upload your own:"""
    file = st.file_uploader(instructions)

    col1, col2 = st.columns(2)

    col1.header("Image")
    if file:
        raw_img = Image.open(file).convert("RGB")
    else:
        raw_img = load_demo_image()

    w, h = raw_img.size
    scaling_factor = 720 / w
    resized_image = raw_img.resize((int(w * scaling_factor), int(h * scaling_factor)))

    col1.image(resized_image, use_column_width=True)
    col2.header("Question")

    user_question = col2.text_input("Input your question!", "Please describe the content in the bbox: {} of the image.")
    if '[' in user_question and ']' in user_question:
        bbox = user_question[user_question.find('['): user_question.find(']') + 1]
    else:
        st.error('Please enter a valid question that must contain bounding box.')
    print(f"user_question: {user_question}")
    print(f"bbox: {bbox}")
    # try:
    #     bbox = [float(num.strip()) for num in bbox.split(',')]
    #     assert len(bbox) == 4, "Bbox must have 4 numbers"
    #     assert all(num >= 0 for num in bbox), "Bbox must have positive numbers"
    #     assert all(num <= 1 for num in bbox), "Bbox must have numbers between 0 and 1"
    # except ValueError:
    #     st.error('Please enter a valid list of numbers')
    user_question = concat_question_bbox(user_question, bbox)
    qa_button = st.button("Submit")

    col2.header("Answer")

    # ===== event =====
    # vis_processor = load_processor("blip_image_eval").build(image_size=480)
    vis_processor = load_processor("blip_image_eval").build(image_size=int(image_size))
    text_processor = load_processor("box_caption_question").build()

    if qa_button:
        print(f"model name: {model_name}, model type: {model_type}")
        if model_base.startswith("BLIP2"):
            model = load_model_cache(
                model_name, model_type=model_type, is_eval=True, device=device
            )

            img = vis_processor(raw_img).unsqueeze(0).to(device)
            # question = text_processor(user_question)
            question = user_question

            vqa_samples = {"image": img, "text_input": [question]}
            # answers = model.predict_answers(vqa_samples, inference_method="generate", max_len=int(max_len), temperature=temperature)
            answers = model.generate({"image": img, "dense_caption": question}, use_nucleus_sampling=True, max_length=int(max_len), temperature=temperature)

            print(f"answers: {answers}")
            col2.write("\n".join(answers))
            
        elif model_base.startswith("BLIP"):
            model = load_model_cache(
                "blip_vqa", model_type="vqav2", is_eval=True, device=device
            )

            img = vis_processor(raw_img).unsqueeze(0).to(device)
            question = text_processor(user_question)

            vqa_samples = {"image": img, "text_input": [question]}
            answers = model.predict_answers(vqa_samples, inference_method="generate", max_len=512)

            print(f"answers: {answers}")
            col2.write("\n".join(answers))
