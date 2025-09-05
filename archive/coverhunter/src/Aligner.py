#!/usr/bin/env python3
# author: liufeng
# datetime: 2023/7/5 5:11 PM

"""coarse-to-fine alignment as mentioned at https://arxiv.org/abs/2306.09025"""

import os

import numpy as np

from CoverHunterMPS.src.utils import dict_to_line, line_to_dict, read_lines, write_lines


class Aligner:
    def __init__(self, embed_dir) -> None:
        self._embed_dir = embed_dir
        self._memory_data = {}
        for name in os.listdir(embed_dir):
            data_path = os.path.join(embed_dir, name)
            embed = np.load(data_path)
            perf_name = name.replace(".npy", "")
            self._memory_data[perf_name] = embed

    def _get_shift_frame(self, data_i, data_j):
        assert data_i["work"] == data_j["work"]
        perf_i = data_i["perf"]
        perf_j = data_j["perf"]
        if perf_i == perf_j:
            return 0

        idx_npy_i = {}
        for k, v in self._memory_data.items():
            if perf_i in k:
                idx_i = int(k.split("▁start-")[1])
                #         # assumes '▁start-' never occurs in an perf code
                idx_npy_i[idx_i] = v

        idx_npy_j = {}
        for k, v in self._memory_data.items():
            if k.startswith(perf_j):
                idx_j = int(k.split("▁start-")[1])
                # assumes '▁start-' never occurs in an perf code
                idx_npy_j[idx_j] = v

        res = []
        for idx_i, npy_i in idx_npy_i.items():
            all_distance = []
            for idx_j, npy_j in idx_npy_j.items():
                if idx_i != idx_j and perf_i != perf_j:
                    all_distance.append((idx_j, np.linalg.norm(npy_i - npy_j)))
            all_distance = sorted(all_distance, key=lambda x: x[1])
            nearest_j = all_distance[0][0]
            res.append((idx_i, nearest_j, idx_i - nearest_j))

        count_delta = {}
        for _, _, z in res:
            if z not in count_delta:
                count_delta[z] = 0
            count_delta[z] += 1
        count_delta_lst = list(count_delta.items())
        count_delta_lst = sorted(count_delta_lst, key=lambda x: x[1], reverse=True)
        shift_frame, _ = count_delta_lst[0]
        return shift_frame

    def align(self, data_path, output_path) -> None:
        dump_lines = []
        data_lines = read_lines(data_path)
        for line_i in data_lines:
            local_data_i = line_to_dict(line_i)
            for line_j in data_lines:
                local_data_j = line_to_dict(line_j)
                if local_data_i["work_id"] == local_data_j["work_id"]:
                    shift_frame = self._get_shift_frame(local_data_i, local_data_j)
                    dump_lines.append(
                        dict_to_line(
                            {
                                "perf_i": local_data_i["perf"],
                                "perf_j": local_data_j["perf"],
                                "shift_frame_i_to_j": shift_frame,
                            },
                        ),
                    )
        write_lines(output_path, dump_lines)
