#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

    def export_changed_args_to_json(self, args):
        defaults = {}
        for arg in vars(args).items():
            try:
                if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                    defaultvalue = getattr(self, arg[0])
                    # defaults[ arg[0] ] = defaultvalue
                    if defaultvalue != arg[1]:
                        defaults[arg[0]] = arg[1]
            except:
                pass

        return defaults


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = 1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.model = "tad_gaussian"
        self.loader = "colmap"

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 45_001
        self.position_lr_init = 0.00026
        self.position_lr_final = 0.0000026
        self.position_lr_delay_mult = 0.1
        self.position_lr_max_steps = 45_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.0015

        self.trbfs_lr = 0.003
        self.trbfslinit = 1.0
        self.batch = 1
        self.movelr = 3.5

        self.omega_lr = 0.0001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2

        # Initialize point cloud every N timestamp
        self.init_pcd_every = 20
        self.evaluate_every_iter = 1000
        self.motion_mode = "fourier_asym"
        self.motion_degree = 4
        self.opacity_mode = "deform_gauss_op_asym"
        self.visibility_threshold = 0.2
        self.min_visible_timestamps = 50
        self.temporal_span = 50
        self.span_budget = 70
        self.alpha = 0.8
        self.beta = 0.75
        self.pcd = "sparse"
        self.pcd_downsample = 1

        self.densification_interval = 200
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 17000
        self.densify_grad_threshold = 0.0002
        self.reg = 0
        self.regl = 0.0001
        self.opthr = 0.005
        self.gtisint8 = 0  # 0: gt used as float, 1: gt kept as int8 for memory
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
