import gradio as gr
import os
from PIL import Image
import subprocess


# check if there is a picture uploaded or selected
def check_img_input(control_image):
    if control_image is None:
        raise gr.Error("Please select or upload an input image")


def optimize_stage_1(image_block: Image.Image, preprocess_chk: bool, elevation_slider: float):
    if not os.path.exists('tmp_data'):
        os.makedirs('tmp_data')
    if preprocess_chk:
        # save image to a designated path
        image_block.save(os.path.join('tmp_data', 'tmp.png'))

        # preprocess image
        print(f'python process.py {os.path.join("tmp_data", "tmp.png")}')
        subprocess.run(f'python process.py {os.path.join("tmp_data", "tmp.png")}', shell=True)
    else:
        image_block.save(os.path.join('tmp_data', 'tmp_rgba.png'))

    # stage 1
    subprocess.run(f'python main.py --config {os.path.join("configs", "image.yaml")} input={os.path.join("tmp_data", "tmp_rgba.png")} save_path=tmp mesh_format=glb elevation={elevation_slider} force_cuda_rast=True', shell=True)

    return os.path.join('logs', 'tmp_mesh.glb')


def optimize_stage_2(elevation_slider: float):
    # stage 2
    subprocess.run(f'python main2.py --config {os.path.join("configs", "image.yaml")} input={os.path.join("tmp_data", "tmp_rgba.png")} save_path=tmp mesh_format=glb elevation={elevation_slider} force_cuda_rast=True', shell=True)

    return os.path.join('logs', 'tmp.glb')


if __name__ == "__main__":
    _TITLE = '''GaussianImage: Generative Gaussian Splatting for Efficient 3D Content Creation'''

    _IMG_USER_GUIDE = "Please upload an image in the block above (or choose an example above) and click **Generate 3D**."

    # load images in 'data' folder as examples
    example_folder = os.path.join(os.path.dirname(__file__), 'data')
    example_fns = os.listdir(example_folder)
    example_fns.sort()
    examples_full = [os.path.join(example_folder, x) for x in example_fns if x.endswith('.png')]

    # Compose demo layout & data flow
    with gr.Blocks(title=_TITLE, theme=gr.themes.Soft()) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('# ' + _TITLE)


        # Image-to-3D
        with gr.Row(variant='panel'):
            with gr.Column(scale=5):
                image_block = gr.Image(type='pil', image_mode='RGBA', height=290, label='Input image')

                elevation_slider = gr.Slider(-90, 90, value=0, step=1, label='Estimated elevation angle')
                gr.Markdown(
                    "default to 0 (horizontal), range from [-90, 90]. If you upload a look-down image, try a value like -30")

                preprocess_chk = gr.Checkbox(True,
                                             label='Preprocess image automatically (remove background and recenter object)')

                gr.Examples(
                    examples=examples_full,  # NOTE: elements must match inputs list!
                    inputs=[image_block],
                    outputs=[image_block],
                    cache_examples=False,
                    label='Examples (click one of the images below to start)',
                    examples_per_page=40
                )
                img_run_btn = gr.Button("Generate 3D")
                img_guide_text = gr.Markdown(_IMG_USER_GUIDE, visible=True)

            with gr.Column(scale=5):
                obj3d_stage1 = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Model (Stage 1)")
                obj3d = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Model (Final)")

            # if there is an input image, continue with inference
            # else display an error message
            img_run_btn.click(check_img_input, inputs=[image_block], queue=False).success(optimize_stage_1,
                                                                                          inputs=[image_block,
                                                                                                  preprocess_chk,
                                                                                                  elevation_slider],
                                                                                          outputs=[
                                                                                              obj3d_stage1]).success(
                optimize_stage_2, inputs=[elevation_slider], outputs=[obj3d])

    demo.queue().launch(share=True)