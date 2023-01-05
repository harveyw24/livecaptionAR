# import cv2
# import tempfile
# from argparse import Namespace

# import fairseq
# from fairseq import tasks, utils
# from fairseq.dataclass.configs import GenerationConfig

import fairseq
from avhubert import hubert_pretraining, hubert
from avhubert import extract_roi

import cv2
import tempfile
from argparse import Namespace
import fairseq
import sys
import pickle
import joblib
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.configs import GenerationConfig
import torch
import os
from os.path import exists
import subprocess
from avhubert import extract_roi

class HubertModel:
    def __init__(self, ckpt_path, wd):
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.models, self.cfg, self.task = models, cfg, task
        self.wd = wd
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

    def transcribe(self, file_name):
        orig_vid_path = self.wd + f'/data/video-orig/{file_name}'
        updated_file_name = os.path.splitext(file_name)[0]
        compress_vid_path = self.wd + f'/data/video-comp/{updated_file_name}_compressed.mp4'
        mouth_roi_path = self.wd[1:] + f'/data/video-roi/{updated_file_name}_roi.mp4'
        audio_path = self.wd + f'/data/audio/{updated_file_name}_audio.wav'

        if not exists(orig_vid_path):
            print(f'File [{orig_vid_path}] does not exist')
            return

        # Preprocess
        if not exists('/' + mouth_roi_path):
            # command = f'ffmpeg -i {orig_vid_path} -vcodec libx265 -crf 28 -r 25 -vf scale=850:480 -ac 1 {compress_vid_path}'
            # subprocess.call(command, shell=True)

            # EXTRACT ROI
            face_predictor_path = f"{self.wd}/data/misc/shape_predictor_68_face_landmarks.dat"
            mean_face_path = f"{self.wd}/data/misc/20words_mean_face.npy"
            extract_roi.preprocess_video(compress_vid_path, '/' + mouth_roi_path, face_predictor_path, mean_face_path)

            # EXTRACT AUDIO
            # command = f"ffmpeg -i {compress_vid_path} -ar 16000 -ac 1 -f wav {audio_path}"
            command = f"ffmpeg -i {orig_vid_path} -ar 16000 -ac 1 -f wav {audio_path}"
            subprocess.call(command, shell=True)


        user_dir = "./"
        newHypo = self.predict(mouth_roi_path, audio_path, user_dir, self.models, self.cfg, self.task)
        print(newHypo)

        with open(f'{self.wd}/data/transcriptions/{updated_file_name}.txt', 'w') as f:
            f.write(f'{file_name} - Speech Transcription\n')
            f.write(newHypo) 

        # Delete original file
        print("Cleaning original video folder")
        os.remove(orig_vid_path)

        return newHypo