from typing import NamedTuple
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import torch
import cv2
import io

import aspose.threed as a3d

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.util.pc_to_mesh import marching_cubes_mesh
from point_e.models.download import load_checkpoint
from point_e.util.plotting import plot_point_cloud
from point_e.util.point_cloud import PointCloud

from flask import Flask, request, Response, send_file#, send_from_directory


app = Flask(__name__)
device = torch.device('cuda')


class Pointcloud(NamedTuple):
	img2pointcloud: str = 'img2pointcloud'
	text2pointcloud: str = 'text2pointcloud'


pointcloud = Pointcloud()


def fig2img(fig):
	buf = io.BytesIO()
	fig.savefig(buf)
	buf.seek(0)
	img = Image.open(buf)
	return img


@app.route('/generate', methods=['OPTIONS', 'POST'])
def generate():
	if (request.method == 'OPTIONS'):
		print('got options 1')
		response = Response()
		response.headers['Access-Control-Allow-Origin'] = '*'
		response.headers['Access-Control-Allow-Headers'] = '*'
		response.headers['Access-Control-Allow-Methods'] = '*'
		response.headers['Access-Control-Expose-Headers'] = '*'
		response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
		response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
		response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
		print('got options 2')
		return response

	fig = None
	pointcloudarg = request.args.get('pointcloud') or None

	# img2pointcloud
	if pointcloudarg == pointcloud.img2pointcloud:
		sampler = PointCloudSampler(
			device=device,
			models=[img2pointcloud_base_model, img2pointcloud_upsampler_model],
			diffusions=[img2pointcloud_base_diffusion, img2pointcloud_upsampler_diffusion],
			num_points=[1024, 4096 - 1024],
			aux_channels=['R', 'G', 'B'],
			guidance_scale=[3.0, 3.0],
		)

		# Load an image to condition on.
		# img = Image.open('example_data/cube_stack.jpg')
		body = request.get_data()
		imencode_img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
		img = Image.fromarray(imencode_img)
		# img = np.asarray(Image.open(body))

		# Produce a sample from the model.
		samples = None
		for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
		    samples = x

		pc = sampler.output_to_point_clouds(samples)[0]
		fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))

	# text2pointcloud
	if pointcloudarg == pointcloud.text2pointcloud:
		sampler = PointCloudSampler(
			device=device,
			models=[text2pointcloud_base_model, text2pointcloud_upsampler_model],
			diffusions=[text2pointcloud_base_diffusion, text2pointcloud_upsampler_diffusion],
			num_points=[1024, 4096 - 1024],
			aux_channels=['R', 'G', 'B'],
			guidance_scale=[3.0, 0.0],
			model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
		)

		# Set a prompt to condition on.
		prompt = request.args.get('prompt')

		# Produce a sample from the model.
		samples = None
		for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
		    samples = x

		pc = sampler.output_to_point_clouds(samples)[0]
		fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))

	if pointcloudarg is None:
		npz = request.files['npz']
		# Load a point cloud we want to convert into a mesh.
		pc = PointCloud.load(npz) # 'point_e/examples/example_data/pc_corgi.npz'

		# Plot the point cloud as a sanity check.
		fig = plot_point_cloud(pc, grid_size=2)

		# Produce a mesh (with vertex colors)
		mesh = marching_cubes_mesh(
			pc=pc,
			model=model,
			batch_size=4096,
			grid_size=32, # increase to 128 for resolution used in evals
			progress=True,
		)

		# Write the mesh to a PLY file to import into some other program.
		# OBS don't save as glb here! transform the ply
		with open('mesh.ply', 'wb') as f:
			mesh.write_ply(f)

		# return send_from_directory('./', 'mesh.ply', as_attachment=True)
		# create scene for transform
		scene = a3d.Scene.from_file('mesh.ply')
		# save new file
		scene.save('mesh.glb')
		return send_file('./mesh.glb', as_attachment=True)

	if fig is not None:
		output = fig2img(fig)
		img_byte_arr = io.BytesIO()
		output.save(img_byte_arr, format='PNG')
		img_byte_arr = img_byte_arr.getvalue()
		response = Response(img_byte_arr, headers={'Content-Type':'image/png'})
		response.headers['Access-Control-Allow-Origin'] = '*'
		response.headers['Access-Control-Allow-Headers'] = '*'
		response.headers['Access-Control-Allow-Methods'] = '*'
		response.headers['Access-Control-Expose-Headers'] = '*'
		response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
		response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
		response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
		return response

	# if pointcloudarg is not pointcloud.img2pointcloud or pointcloud.text2pointcloud:
	# 	message = 'the provided pointcloud argument does not exists or is not supported'


if __name__ == '__main__':
	print('creating img2pointcloud base model...')
	img2pointcloud_base_name = 'base40M' # use base300M or base1B for better results
	img2pointcloud_base_model = model_from_config(MODEL_CONFIGS[img2pointcloud_base_name], device)
	img2pointcloud_base_model.eval()
	img2pointcloud_base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[img2pointcloud_base_name])

	print('creating img2pointcloud upsample model...')
	img2pointcloud_upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
	img2pointcloud_upsampler_model.eval()
	img2pointcloud_upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

	print('downloading img2pointcloud base checkpoint...')
	img2pointcloud_base_model.load_state_dict(load_checkpoint(img2pointcloud_base_name, device))

	print('downloading img2pointcloud upsampler checkpoint...')
	img2pointcloud_upsampler_model.load_state_dict(load_checkpoint('upsample', device))

	print('creating text2pointcloud base model...')
	text2pointcloud_base_name = 'base40M-textvec'
	text2pointcloud_base_model = model_from_config(MODEL_CONFIGS[text2pointcloud_base_name], device)
	text2pointcloud_base_model.eval()
	text2pointcloud_base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[text2pointcloud_base_name])

	print('creating text2pointcloud upsample model...')
	text2pointcloud_upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
	text2pointcloud_upsampler_model.eval()
	text2pointcloud_upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

	print('downloading text2pointcloud base checkpoint...')
	text2pointcloud_base_model.load_state_dict(load_checkpoint(text2pointcloud_base_name, device))

	print('downloading text2pointcloud upsampler checkpoint...')
	text2pointcloud_upsampler_model.load_state_dict(load_checkpoint('upsample', device))

	print('creating SDF model...')
	name = 'sdf'
	model = model_from_config(MODEL_CONFIGS[name], device)
	model.eval()

	print('loading SDF model...')
	model.load_state_dict(load_checkpoint(name, device))

	app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
