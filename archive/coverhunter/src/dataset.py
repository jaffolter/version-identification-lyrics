#!/usr/bin/env python3

import logging
import math
import random
import subprocess
import os

import librosa
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.sampler import Sampler
from scipy.ndimage import uniform_filter1d

from CoverHunterMPS.src.cqt import shorter
from CoverHunterMPS.src.utils import line_to_dict, read_lines

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s"
)

# set fixed seeding for dataset classes
# test of instead using hyperparameter seed throughout all these operations
# did NOT show benefits to either learning, and this approach still resulted in
# deterministic training at least on MPS (4 November 2024)

random.seed(1)


def _float_signal_to_int16(signal):
    signal = signal * 32768
    signal = np.int16(signal)
    return signal


class SignalAug:
    """Signal Augmentation, not change signal shape except "_change_speed" function

    Notes:
      input should be numpy.array of int16[-32768, 32767] or float[0, 1]
      output will be numpy.array float[0, 1]
    """

    def __init__(self, hp) -> None:
        self._hp = hp
        if "add_noise" in self._hp:
            self._noise_path_lst = read_lines(
                hp["add_noise"]["noise_path"], log=False
            )
            logging.info(
                "Noise Data path:{}".format(hp["add_noise"]["noise_path"])
            )
            logging.info(f"Noise Data items:{len(self._noise_path_lst)}")
        else:
            self._noise_path_lst = None
        if "seed" not in hp:
            hp["seed"] = 1234
        random.seed(hp["seed"])
        np.random.seed(hp["seed"])
        logging.info(f"SignalAug hparams: {hp}")

    @staticmethod
    def _change_volume(signal, coef):
        """change signal magnitude with coef

        Args:
          signal:
          coef: float from 0 to 1

        Returns:

        """
        max_val = np.max(np.abs(signal)) + 0.01
        signal = signal / max_val * 0.999
        return signal * coef

    @staticmethod
    def _add_noise(signal, noise, snr):
        """add noise to signal with snr

        Args:
          signal: numpy
          noise: numpy
          snr: float, log-signal-to-noise-ratio, smaller means more noise

        Returns:
          noise

        """

        signal = signal / max(0.001, np.max(np.abs(signal))) * 0.495
        noise = noise / max(0.001, np.max(np.abs(noise))) * 0.495
        snr = 10 ** (snr / 10.0)
        coef = np.sqrt(1 / snr)
        new_signal = signal + noise * coef
        return new_signal / max(0.001, np.max(np.abs(new_signal))) * 0.95

    @staticmethod
    def _change_tempo(signal, coef):
        """

        Args:
          signal:
          coef:

        Returns:

        Notes:
          length of signal will be changed
        """
        args = [
            "sox",
            "-t",
            "s16",
            "-r",
            "16000",
            "-b",
            "16",
            "-c",
            "1",
            "-",
            "-r",
            "16000",
            "-t",
            "raw",
            "-",
            "tempo",
            "-s",
            coef,
        ]
        args = [str(x) for x in args]
        logging.info("cmd:{}".format(" ".join(args)))

        process_handle = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        x = np.int16(signal)
        out, err = process_handle.communicate(x.T.tobytes(order="F"))
        out = np.frombuffer(out, dtype=np.int16)

        status = process_handle.returncode
        if status > 0:
            logging.info(
                "status:{}, err:{}".format(status, err.decode("utf-8"))
            )

        return out

    @staticmethod
    def _change_speed(signal, coef):
        """

        Args:
          signal:
          coef:

        Returns:

        Notes:
          length of signal will be changed
        """
        args = [
            "sox",
            "-t",
            "s16",
            "-r",
            "16000",
            "-b",
            "16",
            "-c",
            "1",
            "-",
            "-r",
            "16000",
            "-t",
            "raw",
            "-",
            "speed",
            coef,
        ]
        args = [str(x) for x in args]
        logging.info("cmd:{}".format(" ".join(args)))

        process_handle = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        x = _float_signal_to_int16(signal)
        out, err = process_handle.communicate(x.T.tobytes(order="F"))
        out = np.frombuffer(out, dtype=np.int16)

        status = process_handle.returncode
        if status > 0:
            logging.info(
                "status:{}, err:{}".format(status, err.decode("utf-8"))
            )
        signal = out
        return signal

    @staticmethod
    def _change_pitch(signal, pitch_factor):
        """wrapper for sox

        Args:
          signal: numpy.array
          pitch_factor: float, means ratio of new pitch / old pitch

        Returns:
          augmented signal

        Notes:
          Very slow

        References:
          https://github.com/rabitt/pysox/blob/master/sox/transform.py
        """
        semitone = math.log(pitch_factor, 2) * 12
        cents_of_semitone = semitone * 100
        args = [
            "sox",
            "-t",
            "s16",
            "-r",
            "16000",
            "-b",
            "16",
            "-c",
            "1",
            "-",
            "-r",
            "16000",
            "-t",
            "raw",
            "-",
            "pitch",
            cents_of_semitone,
        ]
        args = [str(x) for x in args]

        process_handle = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        x = _float_signal_to_int16(signal)
        out, err = process_handle.communicate(x.T.tobytes(order="F"))
        if process_handle.returncode != 0:
            return signal
        out = np.frombuffer(out, dtype=np.int16)
        # print("status:", status)
        # print("err:", err.decode("utf-8"))
        return out

    def augmentation(self, signal):
        """vanilla signal[N] -> augmentation signal[N]"""
        signal = signal.astype(float)
        if "speed" in self._hp:
            if random.random() <= self._hp["speed"]["prob"]:
                coef = random.choice(self._hp["speed"]["coef"])
                signal = self._change_speed(signal, coef)

        if "tempo" in self._hp:
            if random.random() <= self._hp["tempo"]["prob"]:
                coef = random.choice(self._hp["tempo"]["coef"])
                signal = self._change_tempo(signal, coef)

        if "pitch" in self._hp:
            if random.random() <= self._hp["pitch"]["prob"]:
                coef = random.choice(self._hp["pitch"]["shift"])
                signal = self._change_pitch(signal, coef)

        if "add_noise" in self._hp and self._hp["add_noise"]["prob"] > 0.01:
            hp_noise = self._hp["add_noise"]
            noise_data = line_to_dict(random.choice(self._noise_path_lst))
            noise_sig_init, _ = librosa.load(
                noise_data["wav"], sr=hp_noise["sr"]
            )
            noise_signal = noise_sig_init[
                random.randint(0, len(noise_sig_init)) :
            ]
            while len(noise_signal) < len(signal):
                noise_signal = np.concatenate([noise_signal, noise_sig_init])

            to_add_noise = np.zeros_like(signal)
            chunk = hp_noise["sr"] * hp_noise["chunk"]
            for i in range(int(len(to_add_noise) / chunk)):
                if random.random() <= hp_noise["prob"]:
                    to_add_noise[i * chunk : (i + 1) * chunk] = noise_signal[
                        i * chunk : (i + 1) * chunk
                    ]

            snr = random.choice([-30, -20, -10, 0, 10])
            signal = self._add_noise(signal, noise=to_add_noise, snr=snr)

        if "volume" in self._hp:
            if random.random() <= self._hp["volume"]["prob"]:
                coef = random.random()
                signal = self._change_volume(signal, coef)
        return signal


class SpecAug:
    """Spectral Augmentation (only shifts or masks data)"""

    def __init__(self, hp=None, logger=None, device=None) -> None:
        self._hp = hp
        self.device = device
        if "seed" not in hp:
            hp["seed"] = 1234
        random.seed(hp["seed"])
        np.random.seed(hp["seed"])
        if logger:
            logger.info(f"SpecAug hparams {hp}")

        # next 4 lines only for visualization of augmentations
        self.save_dir = "augmentation_visualization"
        os.makedirs(self.save_dir, exist_ok=True)
        self.count = 0
        self.current_perf = (
            None  # New attribute to store the current perf value
        )

    def _save_spectrograms(self, original, augmented, method):
        # only used for visualization of low_melody augmentation
        # not safe for MPS when using multiprocessing 'fork' method
        # because of matplotlib problems
        import matplotlib.pyplot as plt

        # Find global min/max for consistent amplitude normalization
        vmin = min(original.min(), augmented.min())
        vmax = max(original.max(), augmented.max())

        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(
            original.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        plt.title("Original Spectrogram")
        plt.subplot(122)
        plt.imshow(
            augmented.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        plt.title(f"Augmented Spectrogram ({method})")
        plt.tight_layout()

        # Use the perf value in the filename
        filename = f"{self.current_perf}_{method}_{self.count:04d}.png"
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()
        self.count += 1

    def _save_spectrograms_mps(self, original, augmented, method):
        """Save spectrograms using PIL with text annotations,
        instead of matplotlib, because PIL is safe for forked processes on MPS
        """
        # only used for visualization of low_melody augmentation
        from PIL import Image, ImageDraw  # , ImageFont

        def normalize_for_display(array):
            """Scale array to 0-255 range for display"""
            array = array - array.min()
            if array.max() > 0:
                array = (array / array.max() * 255).astype(np.uint8)
            return array

        # Normalize arrays for display and flip vertically ([::-1] on first dimension after transpose)
        original_img = normalize_for_display(original.T[::-1])
        augmented_img = normalize_for_display(augmented.T[::-1])

        # Create side-by-side image with space for text
        text_height = 30  # pixels for text
        width = original_img.shape[1] + augmented_img.shape[1]
        height = (
            max(original_img.shape[0], augmented_img.shape[0]) + text_height
        )
        combined = (
            np.zeros((height, width), dtype=np.uint8) + 255
        )  # white background

        # Copy images into combined array below text area
        combined[
            text_height : text_height + original_img.shape[0],
            : original_img.shape[1],
        ] = original_img
        combined[
            text_height : text_height + augmented_img.shape[0],
            original_img.shape[1] :,
        ] = augmented_img

        # Convert to PIL Image for text addition
        img = Image.fromarray(combined)
        draw = ImageDraw.Draw(img)

        # Add text (using default font)
        draw.text((10, 5), "Original", fill=0)  # black text
        draw.text(
            (original_img.shape[1] + 10, 5), f"Augmented ({method})", fill=0
        )

        # Save
        filename = f"{self.current_perf}_{method}_{self.count:04d}.png"
        img.save(os.path.join(self.save_dir, filename))
        self.count += 1

    @staticmethod
    def _mask_silence(feat, p_threshold=0.1):
        if random.random() > p_threshold:
            return feat

        w, h = np.shape(feat)
        feat_aug = feat
        for i in range(w):
            if random.random() < 0.1:
                feat_aug[i, :] = 0
        for i in range(h):
            if random.random() < 0.1:
                feat_aug[:, i] = 0
        return feat

    @staticmethod
    def _duplicate(feat, p_threshold=0.1):
        if random.random() > p_threshold:
            return feat

        w, h = np.shape(feat)
        feat_aug = feat
        for i in range(1, w):
            if random.random() < 0.1:
                feat_aug[i, :] = feat_aug[i - 1, :]
        for i in range(1, h):
            if random.random() < 0.1:
                feat_aug[:, i] = feat_aug[:, i - 1]
        return feat

    @staticmethod
    def _roll(feat, shift_num=12):
        """
        This original CoverHunter method - improved here for efficiency - wraps
        the spectrogram around, copying overflow content to the bottom or top
        of the pitch dimension.

        Args:
        feat (np.array): The input spectrogram of shape (time_steps, frequency_bins).
        shift_num (int): The maximum number of frequency bins to shift.

        Returns:
        np.array: The pitch-rolled spectrogram.
        """
        time_steps, freq_bins = np.shape(feat)
        shift_amount = random.randint(-1, 1) * shift_num
        return np.roll(feat, shift_amount, axis=1)

    def _shift_melody_low(self, feat, shift_num=12):
        """
        Alternative to original _roll() method, better for strongly
        melodic music that appears at the bottom of the CQT frequency range.

        Shift the spectrogram upwards between 1 and shift_num frequency bins.
        Below that, fill with noise that matches the spectral characteristics
        of this spectrogram.

        Args:
        self: only used if generating visualization filenames
        feat (np.array): The input spectrogram of shape (time_steps, frequency_bins).
        shift_num (int): The maximum number of frequency bins to shift.

        Returns:
        np.array: The pitch-transposed spectrogram.
        """
        time_steps, freq_bins = np.shape(feat)
        shift_amount = random.randint(1, shift_num)

        if shift_amount >= freq_bins:
            return feat

        # Uncomment this line if you want to generate visualizations
        # original_feat = feat.copy()

        # Chunking pads with -100 on the fly, so don't fill with noise
        # below the shift in that time region.
        # Create mask for padded regions (where all frequencies are -100)
        is_padded = np.all(feat == -100, axis=1)

        # Find amplitude of quietest frequency bin, excluding padding
        freq_means = np.mean(feat[~is_padded], axis=0)
        target_amplitude = np.min(freq_means)

        # Shift the spectrogram upwards in place
        feat[:, shift_amount:] = feat[:, : freq_bins - shift_amount]

        # Analyze the region about one fifth above where we'll fill
        reference_region = feat[
            ~is_padded, shift_amount : shift_amount + 6 + random.randint(0, 2)
        ]
        unpadded_len = reference_region.shape[0]

        # Calculate local statistics per time step
        local_means = np.mean(reference_region, axis=1).reshape(-1, 1)
        local_stds = np.std(reference_region, axis=1).reshape(-1, 1)

        # Generate base noise matching the local statistics
        noise = np.random.normal(
            loc=local_means,
            scale=local_stds * 0.5,  # Reduced variation
            size=(unpadded_len, shift_amount),
        )

        ## clip noise with time-bin statistics
        noise = np.maximum(noise, local_means - local_stds)  # valleys

        # fade to quiet towards bottom frequencies
        if shift_amount > 2:
            # Create weights that fade from 1.0 to this quietest level
            #            fade_weights = np.linspace(0, 1, shift_amount).reshape(1, -1)
            fade_weights = 0.5 * (
                1 - np.cos(np.linspace(0, np.pi, shift_amount))
            ).reshape(1, -1)
            noise = noise - (local_means - target_amplitude) * (
                1 - fade_weights
            )

        # clip peaks
        noise = np.minimum(noise, local_means + local_stds)  # peaks

        # Create smooth transition at the boundary by mirroring content
        transition_width = min(5, shift_amount)
        # set weights for far-to-near rows from boundary
        fade = np.linspace(0.1, 0.6, transition_width)
        noise[:, -transition_width:] = (
            # Use reference content proportional to fade value
            fade
            * reference_region[:, :transition_width][
                :, ::-1
            ]  # flip to mirror content near boundary
            +
            # Use noise proportional to (1-fade) value
            (1 - fade) * noise[:, -transition_width:]
        )

        # Apply the processed noise to empty region
        feat[~is_padded, :shift_amount] = noise

        #  Fill padded regions with -100
        feat[is_padded, :shift_amount] = -100

        # Uncomment these lines if you want to generate visualizations
        # CAUTION: If you are using an MPS device, you must also temporarily
        # disable the line "mp.set_start_method("fork")" in whichever
        # tools.train_... script you are using. Otherwise matplotlib will
        # cause a crash.
        # if not np.array_equal(original_feat, feat):
        #    print(f"augmented {self.current_perf} by {shift_amount}")
        #    self._save_spectrograms(original_feat, feat, "shift_melody_low")

        return feat

    @staticmethod
    def _shift_melody_flex(feat, max_shift=12):
        """
        NOT RECOMMENDED - NEEDS INNOVATIONS FROM NEW MELODY_LOW()

        Alternative to _shift_melody_low() method, better for strongly
        melodic music that only sometimes appears at the bottom or top of the
        CQT frequency range. However, if loud non-melodic content is present
        then this method will be less reliable.

        Detect the tonal center by dominant amplitude, and use that to shift
        the spectrogram up or down by a random amount within a range that is
        not likely to cut off melodic content.

        Fill missing data at the bottom or top with -128 (very low amplitude)
        and discard content that spills outside the CQT frequency range.

        Args:
        feat (np.array): The input spectrogram of shape (time_steps, frequency_bins).
        max_shift (int): The maximum number of frequency bins to shift.

        Returns:
        np.array: The pitch-transposed spectrogram.
        """
        time_steps, freq_bins = np.shape(feat)

        # Estimate the tonal center
        loudness_threshold = np.percentile(
            feat, 90
        )  # Consider top 10% as "loud"
        loud_bins = np.where(feat > loudness_threshold)
        if len(loud_bins[1]) > 0:  # index 1 is frequency dimension
            tonal_center = int(np.mean(loud_bins[1]))
        else:
            tonal_center = freq_bins // 2  # Default to middle if no loud bins

        # Calculate available shift range
        max_down_shift = min(max_shift, tonal_center)
        max_up_shift = min(max_shift, freq_bins - tonal_center)

        # Choose a random shift within the available range
        shift_amount = random.randint(-max_down_shift, max_up_shift)

        # Perform the shift
        if shift_amount > 0:
            feat[:, shift_amount:] = feat[:, :-shift_amount]
            feat[:, :shift_amount] = -128  # Fill with silence
        elif shift_amount < 0:
            feat[:, :shift_amount] = feat[:, -shift_amount:]
            feat[:, shift_amount:] = -128  # Fill with silence

        return feat

    @staticmethod
    def _random_erase(
        feat, region_num=4, region_size=(0.25, 0.1), region_val=-128
    ):
        """
        - region_size(width, height) in percentage of feat size
        - region_val = new value of erased region, where 0 is max loud
        and -128 is max quiet
        """
        w, h = np.shape(feat)
        region_w = int(w * region_size[0])
        region_h = int(h * region_size[1])
        for _ in range(region_num):
            center_w = int(random.random() * (w - region_w))
            center_h = int(random.random() * (h - region_h))
            feat[
                center_w - region_w // 2 : center_w + region_w // 2,
                center_h - region_h // 2 : center_h + region_h // 2,
            ] = region_val
        return feat

    def augmentation(self, feat):
        """vanilla feature[80 x N] -> augmentation feature[80 x N]"""
        if "roll_pitch" in self._hp:
            p = random.random()
            if p <= self._hp["roll_pitch"]["prob"]:
                method = self._hp["roll_pitch"].get("method", "default")
                shift_num = self._hp["roll_pitch"]["shift_num"]

                if method == "default":
                    feat = self._roll(feat, shift_num=shift_num)
                elif method == "low_melody":
                    feat = self._shift_melody_low(feat, shift_num=shift_num)
                elif method == "flex_melody":
                    feat = self._shift_melody_flex(feat, max_shift=shift_num)
                else:
                    raise ValueError(f"Unknown roll_pitch method: {method}")

        if "random_erase" in self._hp:
            p = random.random()
            if p <= self._hp["random_erase"]["prob"]:
                feat = self._random_erase(
                    feat,
                    region_val=random.random() * (-128),
                    region_num=self._hp["random_erase"]["erase_num"],
                    region_size=self._hp["random_erase"]["region_size"],
                )
        return feat


class AudioFeatDataset(torch.utils.data.Dataset):
    """Simple DataSet, to get chunk data one by one.
    If feat length is less than chunk_len,
    it will be padded with zeros.

    train: if true, then SpecAug the sample.
    mode:
      - random: cut chunk from feat from random start
      - defined: cut feat with "start/chunk_len" info from line
    """

    def __init__(
        self,
        hp,
        data_path=None,
        data_lines=None,
        train=False,
        mode="random",
        chunk_len=None,
        logger=None,
    ) -> None:

        if data_path:
            data_lines = read_lines(data_path, log=False)

        self._hp = hp
        self._data = []
        self._mode = mode

        assert mode in ["random", "defined"], f"invalid mode: {mode}"
        if mode == "random":
            for line in data_lines:
                local_data = line_to_dict(line)
                self._data.append(
                    (
                        local_data["perf"],
                        local_data["work_id"],
                        local_data["feat"],
                        local_data["feat_len"],
                        chunk_len,
                    ),
                )
        elif mode == "defined":
            for line in data_lines:
                local_data = line_to_dict(line)
                if "start" not in local_data:
                    local_data["start"] = 0
                self._data.append(
                    (
                        local_data["perf"],
                        local_data["work_id"],
                        local_data["feat"],
                        local_data["start"],
                        chunk_len,
                    ),
                )
        else:
            msg = "invalid mode".format()
            raise Exception(msg)

        if logger:
            logger.info(
                f"Init AudioFeatDataset with mode-{mode}, chunk_len-{chunk_len}",
            )
            logger.info(
                f"Input dataset items: {len(data_lines)}, valid items: {self.__len__()}",
            )

        if train and "spec_augmentation" in hp:
            self._aug = SpecAug(
                hp["spec_augmentation"], logger, device=hp["device"]
            )
        else:
            self._aug = None
            if logger:
                logger.info("No spec_augmentation!")

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx):
        perf, label, feat_path, len_or_start, chunk_len = self._data[idx]

        if self._mode == "random":
            feat_len = len_or_start
            if (feat_len - chunk_len) > 0:
                start = int(random.random() * (feat_len - chunk_len))
            else:
                start = 0
        elif self._mode == "defined":
            perf, label, feat_path, start, chunk_len = self._data[idx]
        else:
            msg = "invalid mode".format()
            raise Exception(msg)

        feat = np.load(feat_path)
        feat = feat[start : start + chunk_len]
        if len(feat) < chunk_len:
            feat = np.pad(
                feat,
                pad_width=((0, chunk_len - len(feat)), (0, 0)),
                mode="constant",
                constant_values=-100,
            )

        feat = shorter(feat, self._hp["mean_size"])
        if self._aug:
            # Set the current perf value, only needed for visualization
            self._aug.current_perf = perf
            feat = self._aug.augmentation(feat)

        feat = torch.from_numpy(feat)
        label = torch.tensor(label).long()
        return perf, feat, label


class MPerClassSampler(Sampler):
    """At every iteration, this will return m samples per class. For example,
    if dataloader's batch-size is 100, and m = 5, then 20 classes with 5
    samples will be returned.

    Supports distributed compute when distribute = True. All samples will be
    distributed across all ranks randomly, but samples with the same label will
    be on the same gpu for contrastive loss.

    Original CoverHunter note for distrbuted use:
      After every epoch, set_epoch should be called, and data index of every
      ranks will be changed. Random of index will accept epoch number as seed,
      to make sure every ranks has not override data.
    """

    def __init__(
        self,
        data_path,
        m,
        batch_size,
        distribute=False,
        logger=None,
        seed=None,
    ) -> None:

        data_lines = read_lines(data_path, log=False)

        self._m_per_class = m
        self._batch_size = batch_size
        self._logger = logger

        self._labels_to_indices = self._get_labels_to_indices(data_lines)
        self._global_labels = list(self._labels_to_indices.keys())

        self.labels = self._global_labels

        assert (
            self._batch_size % self._m_per_class
        ) == 0, "m_per_class must divide batch_size without any remainder"

        self._sample_length = self._get_sample_length()

        if logger:
            logger.info(
                f"Init Sampler with Mper with {self._sample_length} items, and m = {m}, batch_num = {self.num_iters()}"
                "\n",
            )

    def __iter__(self):
        idx_list = [0] * self._sample_length
        i = 0
        num_iters = self.num_iters()
        for _ in range(num_iters):
            random.shuffle(self.labels)
            curr_label_set = self.labels[
                : self._batch_size // self._m_per_class
            ]
            for label in curr_label_set:
                t = self._labels_to_indices[label].copy()
                random.shuffle(t)
                idx_list[i : i + self._m_per_class] = t[: self._m_per_class]
                i += self._m_per_class
        return iter(idx_list)

    def num_iters(self):
        return self._sample_length // self._batch_size

    def _get_sample_length(self):
        sample_length = sum(
            [len(self._labels_to_indices[k]) for k in self.labels]
        )
        sample_length -= sample_length % self._batch_size
        return sample_length

    def _split_label_randoms(self, seed):
        split_label = []
        global_label = list(self._labels_to_indices.keys()).copy()
        random.Random(seed).shuffle(global_label)
        split_label.append(global_label)

        return split_label

    # @staticmethod
    def _get_labels_to_indices(self, data_lines, distribute=False):
        """Creates _labels_to_indices, which is a dictionary mapping each label
        to list of indices that will be used to index into dataset.

        Notes: sum of indices must be equal to len(dataset)
        """
        labels_to_indices = {}
        for index, line in enumerate(data_lines):
            local_data = line_to_dict(line)
            label = local_data["work_id"]

            if label not in labels_to_indices:
                labels_to_indices[label] = []

            labels_to_indices[label].append(index)

        for k in labels_to_indices:
            expand_indices = labels_to_indices[k].copy()
            while len(expand_indices) < self._m_per_class:
                expand_indices.extend(labels_to_indices[k])
            labels_to_indices[k] = expand_indices.copy()
        return labels_to_indices

    def __len__(self) -> int:
        return self._sample_length
