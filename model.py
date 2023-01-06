import cv2
import tempfile
from argparse import Namespace
import dlib
import os
import shutil
import torch
from os.path import exists
import subprocess
import numpy as np

import fairseq
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.configs import GenerationConfig
from avhubert import extract_roi

class HubertModel:
    def __init__(self, wd = os.getcwd(), ckpt_path = None, predictor_path = None, mean_face_path = None):
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([wd + ckpt_path])
        self.models, self.cfg, self.task = models, cfg, task
        self.wd = wd
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(wd + predictor_path)
        self.mean_face = np.load(wd + mean_face_path)
        print("Successfuly loaded model")
    
    def predict(self, video_path, audio_path, user_dir, models, cfg, task):
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
        
        # models = [model.eval().cuda() for model in models]
        with torch.no_grad():
            mods = []
            for model in models:
                mods.append(model.cuda().eval())
                # model.cpu()
                torch.cuda.empty_cache()
        models = mods
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

    def get_landmark(self, base64imgdata):
        image_arr = np.frombuffer(base64imgdata, dtype=np.uint8)
        img = cv2.imdecode(image_arr, flags=cv2.IMREAD_COLOR)
        coords = extract_roi.find_landmark(img, self.detector, self.predictor)
        return coords

    def get_transcription(self, file_name, file_full_name, landmarks):
        images_dir = f'{self.wd}/data/temp/{file_name}'
        orig_vid_path = self.wd + f'/data/video-orig/{file_full_name}'
        compress_vid_path = self.wd + f'/data/video-comp/{file_name}_compressed.mp4'
        mouth_roi_path = self.wd[1:] + f'/data/video-roi/{file_name}_roi.mp4'
        audio_path = self.wd + f'/data/audio/{file_name}_audio.wav'

        if not exists(images_dir):
            print(f'Folder [{images_dir}] does not exist')
            return

        # Generate compressed video from input images and clean up
        command = f'ffmpeg -r 25 -i {images_dir}/%d.jpg -vf scale=850:480 {compress_vid_path}'
        subprocess.call(command, shell=True)

        # Extract ROI
        extract_roi.crop_video(compress_vid_path, f'/{mouth_roi_path}', landmarks, self.mean_face)

        # Extract audio
        command = f"ffmpeg -i {orig_vid_path} -ar 16000 -ac 1 -f wav {audio_path}"
        subprocess.call(command, shell=True)

        # Make prediction
        user_dir = "./"
        newHypo = self.predict(mouth_roi_path, audio_path, user_dir, self.models, self.cfg, self.task)
        print(newHypo)

        # Save prediction to text file
        with open(f'{self.wd}/data/transcriptions/{file_name}.txt', 'w') as f:
            f.write(f'{file_name} - Speech Transcription\n')
            f.write(newHypo) 

        # Delete original file
        print("Cleaning original video folder")
        os.remove(orig_vid_path)

        try:
            shutil.rmtree(images_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        return newHypo