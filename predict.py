'''
Author: Harvey
Date: 2022-11-23 14:27:52
LastEditors: Harvey
LastEditTime: 2022-11-24 00:10:41
Description: 
'''

import fairseq
from avhubert import hubert_pretraining, hubert
from avhubert import extract_roi

# ckpt_path = "./data/base_noise_pt_noise_ft_30h.pt"
# models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
# model = models[0]

# extract_roi.test_create_file('./test_data')

face_predictor_path = "./data/misc/shape_predictor_68_face_landmarks.dat"
mean_face_path = "./data/misc/20words_mean_face.npy"
origin_clip_path = "./data/orig_clip.mp4"
mouth_roi_path = "./data/roi.mp4"
# extract_roi.preprocess_video(origin_clip_path, mouth_roi_path, face_predictor_path, mean_face_path)

# extract_roi.play_video('./data/orig_clip.mp4')
# extract_roi.play_video('./data/roi.mp4')

import cv2
import tempfile
from argparse import Namespace
import fairseq
import sys
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.configs import GenerationConfig
import torch
import os
from os.path import exists
from avhubert import extract_roi

def predict(video_path, ckpt_path, user_dir):
  num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
  data_dir = tempfile.mkdtemp()
  tsv_cont = ["/\n", f"test-0\t{video_path}\t{None}\t{num_frames}\t{int(16_000*num_frames/25)}\n"]
  label_cont = ["DUMMY\n"]
  with open(f"{data_dir}/test.tsv", "w") as fo:
    fo.write("".join(tsv_cont))
  with open(f"{data_dir}/test.wrd", "w") as fo:
    fo.write("".join(label_cont))
  utils.import_user_module(Namespace(user_dir=user_dir))
  modalities = ["video"]
  gen_subset = "test"
  gen_cfg = GenerationConfig(beam=20)
  models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
  # models = [model.eval().cuda() for model in models]
  with torch.no_grad():
    mods = []
    for model in models:
      mods.append(model.cuda().eval())
      # model.cpu()
      torch.cuda.empty_cache()
  models = mods
  saved_cfg.task.modalities = modalities
  saved_cfg.task.data = data_dir
  saved_cfg.task.label_dir = data_dir
  task = tasks.setup_task(saved_cfg.task)
  task.cfg.noise_wav = None       # TODO: edit
  # task.cfg.noise_prob = 0
  task.load_dataset(gen_subset, task_cfg=saved_cfg.task)
  generator = task.build_generator(models, gen_cfg)

  def decode_fn(x):
      dictionary = task.target_dictionary
      symbols_ignore = generator.symbols_to_strip_from_output
      symbols_ignore.add(dictionary.pad())
      return task.datasets[gen_subset].label_processors[0].decode(x, symbols_ignore)

  itr = task.get_batch_iterator(dataset=task.dataset(gen_subset)).next_epoch_itr(shuffle=False)
  sample = next(itr)
  sample = utils.move_to_cuda(sample)
  hypos = task.inference_step(generator, models, sample)
  ref = decode_fn(sample['target'][0].int().cpu())
  hypo = hypos[0][0]['tokens'].int().cpu()
  hypo = decode_fn(hypo)
  return hypo

def newPredict(video_path, audio_path, user_dir, models, cfg, task):
  num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
  data_dir = tempfile.mkdtemp()
  tsv_cont = ["/\n", f"test-0\t{video_path}\t{audio_path}\t{num_frames}\t{int(16_000*num_frames/25)}\n"]
  label_cont = ["DUMMY\n"]
  with open(f"{data_dir}/test.tsv", "w") as fo:
    fo.write("".join(tsv_cont))
  with open(f"{data_dir}/test.wrd", "w") as fo:
    fo.write("".join(label_cont))
  utils.import_user_module(Namespace(user_dir=user_dir))
  modalities = ["audio", "video"]
  gen_subset = "test"
  gen_cfg = GenerationConfig(beam=20)
  cfg.task.modalities = modalities
  cfg.task.data = data_dir
  cfg.task.label_dir = data_dir
  cfg.task.max_sample_size = 1000
  task = tasks.setup_task(cfg.task)
  task.cfg.noise_wav = None # TODO: edit
  task.cfg.noise_prob = 0 # TODO: edit
  task.load_dataset(gen_subset, task_cfg=cfg.task)
  models = [model.eval().cuda() for model in models]
  generator = task.build_generator(models, gen_cfg)

  def decode_fn(x):
      dictionary = task.target_dictionary
      symbols_ignore = generator.symbols_to_strip_from_output
      symbols_ignore.add(dictionary.pad())
      return task.datasets[gen_subset].label_processors[0].decode(x, symbols_ignore)

  itr = task.get_batch_iterator(dataset=task.dataset(gen_subset)).next_epoch_itr(shuffle=False)
  sample = next(itr)
  sample = utils.move_to_cuda(sample)
  hypos = task.inference_step(generator, models, sample)
  ref = decode_fn(sample['target'][0].int().cpu())
  hypo = hypos[0][0]['tokens'].int().cpu()
  hypo = decode_fn(hypo)
  return hypo



user_dir = "./"
ckpt_path = "./data/large_noise_pt_noise_ft_433h.pt"


# orig_vid_path = "/home/harveyw/harveyw/iw/av_hubert_updated/av_hubert/av_hubert/data/peter_clip.mp4"
# compress_vid_path = "/home/harveyw/harveyw/iw/av_hubert_updated/av_hubert/av_hubert/data/peter_clip_compressed.mp4"
# mouth_roi_path = "home/harveyw/harveyw/iw/av_hubert_updated/av_hubert/av_hubert/data/peter_clip_roi.mp4"
# audio_path = "/home/harveyw/harveyw/iw/av_hubert_updated/av_hubert/av_hubert/data/peter_audio.wav"

# video_path = "home/harveyw/harveyw/iw/av_hubert_updated/av_hubert/av_hubert/data/compressed_roi.mp4"


if len(sys.argv) != 2:
  print("Argument needed! Please add the name of your video file, e.g. 'harvey_clip' from harvey_clip.mp4")
  sys.exit()
file_name = sys.argv[1]
wd = os.getcwd()

orig_vid_path = wd + f'/data/{file_name}.mp4'
compress_vid_path = wd + f'/data/{file_name}_compressed.mp4'
mouth_roi_path = wd[1:] + f'/data/{file_name}_roi.mp4'
audio_path = wd + f'/data/{file_name}_audio.wav'


from os.path import exists
import subprocess

file_name = orig_vid_path[orig_vid_path.rfind('/') + 1:]

if not exists('/' + mouth_roi_path):
  command = f'ffmpeg -i {orig_vid_path} -vcodec libx265 -crf 28 -r 25 -vf scale=850:480 -ac 1 {compress_vid_path}'
  subprocess.call(command, shell=True)

  # EXTRACT ROI
  face_predictor_path = "/home/harveyw/harveyw/iw/av_hubert_updated/av_hubert/av_hubert/data/misc/shape_predictor_68_face_landmarks.dat"
  mean_face_path = "/home/harveyw/harveyw/iw/av_hubert_updated/av_hubert/av_hubert/data/misc/20words_mean_face.npy"
  extract_roi.preprocess_video(compress_vid_path, '/' + mouth_roi_path, face_predictor_path, mean_face_path)

  # EXTRACT AUDIO
  command = f"ffmpeg -i {compress_vid_path} -ar 16000 -ac 1 -f wav {audio_path}"
  subprocess.call(command, shell=True)

# COMPRESS VIDEO
# ffmpeg -i andrew_clip.mp4 -vcodec libx265 -crf 28 -r 25 -vf scale=850:480 -ac 1 andrew_compress.mp4


# import subprocess
# EXTRACT AUDIO
# command = f"ffmpeg -i {orig_vid_path} -ar 16000 -ac 1 -f wav {audio_path}"
# subprocess.call(command, shell=True)

models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
newHypo = newPredict(mouth_roi_path, audio_path, user_dir, models, cfg, task)
print(newHypo)

with open(f'{file_name}.txt', 'w') as f:
  f.write(f'{file_name}.mp4 Audio transcription\n')
  f.write(newHypo)
  
# hypo = predict(mouth_roi_path, ckpt_path, user_dir)
# print(hypo)

# face_predictor_path = "/home/harveyw/harveyw/iw/av_hubert_updated/av_hubert/av_hubert/data/misc/shape_predictor_68_face_landmarks.dat"
# mean_face_path = "/home/harveyw/harveyw/iw/av_hubert_updated/av_hubert/av_hubert/data/misc/20words_mean_face.npy"
# extract_roi.preprocess_video(orig_vid_path, "data/compressed_roi.mp4", face_predictor_path, mean_face_path)





# compress video
# ffmpeg -i andrew_clip.mp4 -vcodec libx265 -crf 28 -r 25 -vf scale=850:480 -ac 1 andrew_compress.mp4