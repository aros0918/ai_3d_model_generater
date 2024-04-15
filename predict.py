import gradio as gr
import os
from PIL import Image
import subprocess
from cog import BasePredictor, Path, Input
import torch

def check_img_input(control_image):
    if control_image is None:
        raise gr.Error("Please select or upload an input image")


def optimize_stage_1():

    # stage 1
    subprocess.run(f'python main.py --config {os.path.join("configs", "image.yaml")} input={os.path.join("tmp_data", "tmp_rgba.png")} save_path=tmp mesh_format=glb elevation=0 force_cuda_rast=True', shell=True)

    return os.path.join('logs', 'tmp_mesh.glb')


def optimize_stage_2():
    # stage 2
    subprocess.run(f'python main2.py --config {os.path.join("configs", "image.yaml")} input={os.path.join("tmp_data", "tmp_rgba.png")} save_path=tmp mesh_format=glb elevation=0 force_cuda_rast=True', shell=True)

    return os.path.join('logs', 'tmp.glb')


class Predictor(BasePredictor):

    def predict(self,
            image: Path = Input(description="Image to generate a 3D object from.",),
            scale: float = Input(description="Factor to scale image by", default=1.5)
    ) -> Path:
        if not os.path.exists('tmp_data'):
            os.makedirs('tmp_data')
        image = Image.open(image)    
        image.save('tmp_data/tmp.png')
        subprocess.run(f'python process.py {os.path.join("tmp_data", "tmp.png")}', shell=True)
        optimize_stage_1()
        optimize_stage_2()
        out_mesh_path = './logs/tmp.glb'
        return Path(out_mesh_path)