import gradio as gr
import cv2
import numpy as np
  
# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    image_new[pad_size:pad_size+image.shape[0], pad_size:pad_size+image.shape[1]] = image
    image = np.array(image_new)

    image_height = image.shape[0]
    image_width = image.shape[1]
    
    #x |-> w/2 + (x - w/2) * scale ,y |-> h/2 + (y - h/2) * scale
    scale_matrix = np.array([[scale, 0, (1-scale)*image_width/2], [0, scale, (1-scale)*image_height/2]], dtype=np.float32)
    scale_matrix = to_3x3(scale_matrix)
    
    #x |-> (x - w/2)cos(theta) - (y - h/2)sin(theta) + w/2 ,y |-> (x - w/2)sin(theta) + (y - h/2)cos(theta) + h/2
    #考虑到读取图片时(0,0)是在左上，y轴镜像之后旋转方向相反，故用-theta代替theta
    theta = rotation * np.pi / 180
    rotation_matrix  = np.array([[np.cos(theta), np.sin(theta), (1-np.cos(theta))*image_width/2 - np.sin(theta)*image_height/2],
                        [-np.sin(theta), np.cos(theta), (1-np.cos(theta))*image_height/2 + np.sin(theta)*image_width/2]],dtype=np.float32)
    rotation_matrix = to_3x3(rotation_matrix)
    
    #x |-> x + delta x ,y |-> y + delta y (由于镜像,应该为-delta y) 
    translation_matrix = np.array([[1, 0, translation_x], [0, 1, -translation_y]], dtype=np.float32)
    translation_matrix = to_3x3(translation_matrix)
    
    #x |-> w - x
    flip_matrix = np.array([[1 - 2 * flip_horizontal, 0, image_width * flip_horizontal], [0, 1, 0]], dtype=np.float32)
    flip_matrix = to_3x3(flip_matrix)

    transformed_image = np.array(image)
    #Transformation Composition :  TRSM
    composition_matrix = scale_matrix @ rotation_matrix @ translation_matrix @ flip_matrix   
    transformed_image = cv2.warpAffine(image, composition_matrix[:2], (image_width, image_height), borderValue=(255, 255, 255))
    
  
    ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）



    return transformed_image

# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch()
