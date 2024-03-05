# https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_applying_machine_learning/sagemaker_pytorch_model_zoo/inference.py

from __future__ import absolute_import

import os
import torch
import io
import logging

from sagemaker_inference import content_types, encoder, errors, utils
from PIL import Image
from torchvision import transforms

INFERENCE_ACCELERATOR_PRESENT_ENV = "SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT"
DEFAULT_MODEL_FILENAME = "model.pkl"

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

VALID_CONTENT_TYPES = (content_types.JSON, content_types.NPY)


#!/usr/bin/env python3
import argparse
import glob
# import logging
# import os
from typing import Any, ClassVar, Dict, List
# import torch

from detectron2.config import CfgNode, get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances
from detectron2.utils.logger import setup_logger

from densepose import add_densepose_config
from densepose.structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose_outputs_vertex import (
    DensePoseOutputsTextureVisualizer,
    DensePoseOutputsVertexVisualizer,
    get_texture_atlases,
)
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)
from densepose.vis.densepose_results_textures import (
    DensePoseResultsVisualizerWithTexture,
    get_texture_atlas,
)
from densepose.vis.extractor import (
    CompoundExtractor,
    DensePoseOutputsExtractor,
    DensePoseResultExtractor,
    create_extractor,
)

from video_capture import VideoCapture
from time import time
# import numpy as np


LOGGER_NAME = "muscle_part"
logger = logging.getLogger(LOGGER_NAME)


class Action:
    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        parser.add_argument(
            "-v",
            "--verbosity",
            action="count",
            help="Verbose mode. Multiple -v options increase the verbosity.",
        )


class InferenceAction(Action):
    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(InferenceAction, cls).add_arguments(parser)
        parser.add_argument("cfg", metavar="<config>", help="Config file")
        parser.add_argument("model", metavar="<model>", help="Model file")
        parser.add_argument("input", metavar="<input>", help="Input data")
        parser.add_argument(
            "--opts",
            help="Modify config options using the command-line 'KEY VALUE' pairs",
            default=[],
            nargs=argparse.REMAINDER,
        )

    @classmethod
    def execute(cls: type, args: argparse.Namespace):
        t0 = time()
        logger.info(f"Loading config from {args.cfg}")
        opts = []
        cfg = cls.setup_config(args.cfg, args.model, args, opts)
        cfg.defrost() # for mac m1 https://github.com/youngwanLEE/vovnet-detectron2/issues/4#issuecomment-592006710
        cfg.MODEL.DEVICE = 'cpu' # for mac m1 https://knowing.net/posts/2021/11/install-detectron2-draft/
        logger.info(f"Loading model from {args.model}")
        predictor = DefaultPredictor(cfg)
        logger.info(f"Loading data from {args.input}")
        file_list = cls._get_input_file_list(args.input)
        if len(file_list) == 0:
            logger.warning(f"No input images for {args.input}")
            return
        elif len(file_list) == 1 and file_list[0].split('.')[-1] in ['mp4', 'avi', 'mov', 'webm', 'mkv']: #video
            context = cls.create_context(args, cfg)
            cls.parse_video(file_list[0], context, predictor)
        else:
            import numpy as np
            context = cls.create_context(args, cfg)
            for file_name in file_list:
                img = read_image(file_name, format="BGR")  # predictor expects BGR image.
                with torch.no_grad():
                    outputs = predictor(img)["instances"]
                    for i in [0, 1, 7, 15]:
                        print(outputs.pred_densepose.fine_segm[0][i])
                    cls.execute_on_outputs(context, {"file_name": file_name, "image": img}, outputs)
            cls.postexecute(context)
        print(f'exec time: {time()-t0:.2f}sec')

    @classmethod
    def parse_video(cls: type, file_name, context, predictor):
        import cv2
        import numpy as np

        print('parse video')
        visualizer = context["visualizer"]
        extractor = context["extractor"]
        cap = VideoCapture(file_name)
        w = int(cap.get(3))
        h = int(cap.get(4))
        fps = float(cap.get(5))
        out = cv2.VideoWriter('densepose.mp4', cv2.VideoWriter_fourcc(*'H264'), fps, (w, h))
        print('frame', end=' ')
        for i in range(86400000):
            ret, frame = cap.read()
            if not ret:
                break

            with torch.no_grad():
                outputs = predictor(frame)["instances"]
                # execute_on_outputs
                data = extractor(outputs)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
                image_vis = visualizer.visualize(frame, data)
                out.write(image_vis)
                # cv2.imshow('frame', image_vis)
                print(i, end=' ', flush=True)

            if cv2.waitKey(1) & 0xFF in [27, ord('q'), ord('Q')]:
                break #ESC

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print('')

    @classmethod
    def setup_config(
        cls: type, config_fpath: str, model_fpath: str, args: argparse.Namespace, opts: List[str]
    ):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_fpath)
        cfg.merge_from_list(args.opts)
        if opts:
            cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = model_fpath
        cfg.freeze()
        return cfg

    @classmethod
    def _get_input_file_list(cls: type, input_spec: str):
        if os.path.isdir(input_spec):
            file_list = [
                os.path.join(input_spec, fname)
                for fname in os.listdir(input_spec)
                if os.path.isfile(os.path.join(input_spec, fname))
            ]
        elif os.path.isfile(input_spec):
            file_list = [input_spec]
        else:
            file_list = glob.glob(input_spec)
        return file_list


class DumpAction(InferenceAction):
    """
    Dump action that outputs results to a pickle file
    """

    COMMAND: ClassVar[str] = "dump"

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(cls.COMMAND, help="Dump model outputs to a file.")
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(DumpAction, cls).add_arguments(parser)
        parser.add_argument(
            "--output",
            metavar="<dump_file>",
            default="results.pkl",
            help="File name to save dump to",
        )

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        image_fpath = entry["file_name"]
        logger.info(f"Processing {image_fpath}")
        result = {"file_name": image_fpath}
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                    extractor = DensePoseResultExtractor()
                elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                    extractor = DensePoseOutputsExtractor()
                result["pred_densepose"] = extractor(outputs)[0]
        context["results"].append(result)

    @classmethod
    def create_context(cls: type, args: argparse.Namespace, cfg: CfgNode):
        context = {"results": [], "out_fname": args.output}
        return context

    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        out_fname = context["out_fname"]
        out_dir = os.path.dirname(out_fname)
        if len(out_dir) > 0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out_fname, "wb") as hFile:
            torch.save(context["results"], hFile)
            logger.info(f"Output saved to {out_fname}")


class ShowAction(InferenceAction):
    """
    Show action that visualizes selected entries on an image
    """

    COMMAND: ClassVar[str] = "show"
    VISUALIZERS: ClassVar[Dict[str, object]] = {
        "dp_contour": DensePoseResultsContourVisualizer,
        "dp_segm": DensePoseResultsFineSegmentationVisualizer,
        "dp_u": DensePoseResultsUVisualizer,
        "dp_v": DensePoseResultsVVisualizer,
        "dp_iuv_texture": DensePoseResultsVisualizerWithTexture,
        "dp_cse_texture": DensePoseOutputsTextureVisualizer,
        "dp_vertex": DensePoseOutputsVertexVisualizer,
        "bbox": ScoredBoundingBoxVisualizer,
    }

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(cls.COMMAND, help="Visualize selected entries")
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(ShowAction, cls).add_arguments(parser)
        parser.add_argument(
            "visualizations",
            metavar="<visualizations>",
            help="Comma separated list of visualizations, possible values: "
            "[{}]".format(",".join(sorted(cls.VISUALIZERS.keys()))),
        )
        parser.add_argument(
            "--min_score",
            metavar="<score>",
            default=0.8,
            type=float,
            help="Minimum detection score to visualize",
        )
        parser.add_argument(
            "--nms_thresh", metavar="<threshold>", default=None, type=float, help="NMS threshold"
        )
        parser.add_argument(
            "--texture_atlas",
            metavar="<texture_atlas>",
            default=None,
            help="Texture atlas file (for IUV texture transfer)",
        )
        parser.add_argument(
            "--texture_atlases_map",
            metavar="<texture_atlases_map>",
            default=None,
            help="JSON string of a dict containing texture atlas files for each mesh",
        )
        parser.add_argument(
            "--output",
            metavar="<image_file>",
            default="outputres.png",
            help="File name to save output to",
        )

    @classmethod
    def setup_config(
        cls: type, config_fpath: str, model_fpath: str, args: argparse.Namespace, opts: List[str]
    ):
        opts.append("MODEL.ROI_HEADS.SCORE_THRESH_TEST")
        opts.append(str(args.min_score))
        if args.nms_thresh is not None:
            opts.append("MODEL.ROI_HEADS.NMS_THRESH_TEST")
            opts.append(str(args.nms_thresh))
        cfg = super(ShowAction, cls).setup_config(config_fpath, model_fpath, args, opts)
        return cfg

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        import cv2
        import numpy as np

        visualizer = context["visualizer"]
        extractor = context["extractor"]
        image_fpath = entry["file_name"]
        logger.info(f"Processing {image_fpath}")
        image = cv2.cvtColor(entry["image"], cv2.COLOR_BGR2GRAY)
        image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        data = extractor(outputs)
        image_vis = visualizer.visualize(image, data)
        entry_idx = context["entry_idx"] + 1
        out_fname = cls._get_out_fname(entry_idx, context["out_fname"])
        out_dir = os.path.dirname(out_fname)
        if len(out_dir) > 0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cv2.imwrite(out_fname, image_vis)
        logger.info(f"Output saved to {out_fname}")
        context["entry_idx"] += 1

        return image_vis

    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        pass

    @classmethod
    def _get_out_fname(cls: type, entry_idx: int, fname_base: str):
        base, ext = os.path.splitext(fname_base)
        return base + ".{0:04d}".format(entry_idx) + ext

    @classmethod
    def create_context(cls: type, args: argparse.Namespace, cfg: CfgNode) -> Dict[str, Any]:
        vis_specs = args.visualizations.split(",")
        visualizers = []
        extractors = []
        for vis_spec in vis_specs:
            texture_atlas = get_texture_atlas(args.texture_atlas)
            texture_atlases_dict = get_texture_atlases(args.texture_atlases_map)
            vis = cls.VISUALIZERS[vis_spec](
                cfg=cfg,
                texture_atlas=texture_atlas,
                texture_atlases_dict=texture_atlases_dict,
            )
            visualizers.append(vis)
            extractor = create_extractor(vis)
            extractors.append(extractor)
        visualizer = CompoundVisualizer(visualizers)
        extractor = CompoundExtractor(extractors)
        context = {
            "extractor": extractor,
            "visualizer": visualizer,
            "out_fname": args.output,
            "entry_idx": 0,
        }
        return context


from sagemaker.serve import InferenceSpec
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T


def model_fn(model_dir: str):
    #set argparse & logger
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    #set default args
    args.cfg = '../densepose_rcnn_R_50_FPN_s1x_legacy.yaml'
    args.model = '../densepose_rcnn_R_50_FPN_s1x_legacy.pkl'
    args.input = '1583.jpeg'#'1min_workout.mp4' 
    args.output = 'muscle_part.png'
    #args for show action
    args.min_score = 0.8
    args.nms_thresh = None
    args.visualizations = 'dp_segm'
    args.texture_atlas = '../texture_atlas_213.png'
    args.texture_atlases_map = None
    global logger
    logger = setup_logger(name=LOGGER_NAME)
    logger.setLevel(logging.INFO)

    #load model
    cls = ShowAction()
    logger.info(f"Loading config from {args.cfg}")
    args.opts = []
    opts = []
    cfg = cls.setup_config(args.cfg, args.model, args, opts)
    cfg.defrost() # for mac m1 https://github.com/youngwanLEE/vovnet-detectron2/issues/4#issuecomment-592006710
    cfg.MODEL.DEVICE = 'cpu' # for mac m1 https://knowing.net/posts/2021/11/install-detectron2-draft/
    logger.info(f"Loading model from {args.model}")

    #from DefaultPredictor
    cfg_tmp = cfg.clone()  # cfg can be modified by model
    model = build_model(cfg_tmp)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    #final
    logger.info(f"Loading data from {args.input}")

    return model

def generate_args():
    #set argparse & logger
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    #set default args
    args.cfg = '../densepose_rcnn_R_50_FPN_s1x_legacy.yaml'
    args.model = '../densepose_rcnn_R_50_FPN_s1x_legacy.pkl'
    args.input = '1583.jpeg'#'1min_workout.mp4' 
    args.output = 'muscle_part.png'
    #args for show action
    args.min_score = 0.8
    args.nms_thresh = None
    args.visualizations = 'dp_segm'
    args.texture_atlas = '../texture_atlas_213.png'
    args.texture_atlases_map = None
    
    #load model
    cls = ShowAction()
    logger.info(f"Loading config from {args.cfg}")
    args.opts = []
    opts = []
    cfg = cls.setup_config(args.cfg, args.model, args, opts)
    cfg.defrost() # for mac m1 https://github.com/youngwanLEE/vovnet-detectron2/issues/4#issuecomment-592006710
    cfg.MODEL.DEVICE = 'cpu' # for mac m1 https://knowing.net/posts/2021/11/install-detectron2-draft/
    
    aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
    
    return args, cfg, cls, aug
    

def predict_fn(input_object: object, model: object):
    t0 = time()
    args, cfg, cls, aug = generate_args()
    #image
    context = cls.create_context(args, cfg)
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        # Apply pre-processing to image.
        if cfg.INPUT.FORMAT == "RGB":
            # whether the model expects BGR inputs or RGB
            input_object = input_object[:, :, ::-1]
        height, width = input_object.shape[:2]
        image = aug.get_transform(input_object).apply_image(input_object)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        image.to(cfg.MODEL.DEVICE)

        inputs = {"image": image, "height": height, "width": width}

        outputs = model([inputs])[0]["instances"]
        # for i in [0, 1, 7, 15]:
        #     print(outputs.pred_densepose.fine_segm[0][i])
        output = cls.execute_on_outputs(context, {"file_name": args.output, "image": input_object}, outputs)

    #video


    # if len(file_list) == 0:
    #     logger.warning(f"No input images for {self.args.input}")
    #     return
    # elif len(file_list) == 1 and file_list[0].split('.')[-1] in ['mp4', 'avi', 'mov', 'webm', 'mkv']: #video
    #     context = self.cls.create_context(self.args, self.cfg)
    #     self.cls.parse_video(file_list[0], context, model)
    # else:
    #     context = self.cls.create_context(self.args, self.cfg)
    #     for file_name in file_list:
    #         img = read_image(file_name, format="BGR")  # predictor expects BGR image.
    #         with torch.no_grad():
    #             outputs = model(img)["instances"]
    #             for i in [0, 1, 7, 15]:
    #                 print(outputs.pred_densepose.fine_segm[0][i])
    #             self.cls.execute_on_outputs(context, {"file_name": file_name, "image": img}, outputs)
    #     self.cls.postexecute(context)
    print(f'exec time: {time()-t0:.2f}sec')
    return output


def input_fn(input_data, content_type):
    return input_data

### Orig. Code ###

class ModelLoadError(Exception):
    pass


def _is_model_file(filename):
    is_model_file = False
    if os.path.isfile(filename):
        _, ext = os.path.splitext(filename)
        is_model_file = ext in [".pt", ".pth", 'pkl']
    return is_model_file


def model_fn_old(model_dir):
    """Loads a model. For PyTorch, a default function to load a model only if Elastic Inference is used.
    In other cases, users should provide customized model_fn() in script.
    Args:
        model_dir: a directory where model is saved.
    Returns: A PyTorch model.
    """
    if os.getenv(INFERENCE_ACCELERATOR_PRESENT_ENV) == "true":
        model_path = os.path.join(model_dir, DEFAULT_MODEL_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Failed to load model with default model_fn: missing file {}.".format(
                    DEFAULT_MODEL_FILENAME
                )
            )
        # Client-framework is CPU only. But model will run in Elastic Inference server with CUDA.
        try:
            return torch.jit.load(model_path, map_location=torch.device("cpu"))
        except RuntimeError as e:
            raise ModelLoadError(
                "Failed to load {}. Please ensure model is saved using torchscript.".format(
                    model_path
                )
            ) from e
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join(model_dir, DEFAULT_MODEL_FILENAME)
        if not os.path.exists(model_path):
            model_files = [file for file in os.listdir(model_dir) if _is_model_file(file)]
            if len(model_files) != 1:
                raise ValueError(
                    "Exactly one .pth or .pt file is required for PyTorch models: {}".format(
                        model_files
                    )
                )
            model_path = os.path.join(model_dir, model_files[0])
        try:
            model = torch.jit.load(model_path, map_location=device)
        except RuntimeError as e:
            raise ModelLoadError(
                "Failed to load {}. Please ensure model is saved using torchscript.".format(
                    model_path
                )
            ) from e
        model = model.to(device)
        return model


def input_fn_old(input_data, content_type):
    """
    Args:
        input_data: the request payload serialized in the content_type format
        content_type: the request content_type
    """
    if content_type == "application/x-image":
        decoded = Image.open(io.BytesIO(input_data))
    else:
        raise ValueError(f"Type [{content_type}] not supported.")

    preprocess = transforms.Compose([transforms.ToTensor()])
    normalized = preprocess(decoded)
    return normalized


def predict_fn_old(data, model):
    """A default predict_fn for PyTorch. Calls a model on data deserialized in input_fn.
    Runs prediction on GPU if cuda is available.
    Args:
        data: input data (torch.Tensor) for prediction deserialized by input_fn
        model: PyTorch model loaded in memory by model_fn
    Returns: a prediction
    """
    with torch.no_grad():
        if os.getenv(INFERENCE_ACCELERATOR_PRESENT_ENV) == "true":
            device = torch.device("cpu")
            model = model.to(device)
            input_data = data.to(device)
            model.eval()
            with torch.jit.optimized_execution(True, {"target_device": "eia:0"}):
                output = model([input_data])
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            input_data = data.to(device)
            model.eval()
            output = model([input_data])

    return output


def output_fn(prediction, accept):
    """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPY format.
    Args:
        prediction: a prediction result from predict_fn
        accept: type which the output data needs to be serialized
    Returns: output data serialized
    """
    if type(prediction) == torch.Tensor:
        prediction = prediction.detach().cpu().numpy().tolist()

    for content_type in utils.parse_accept(accept):
        if content_type in encoder.SUPPORTED_CONTENT_TYPES:
            encoded_prediction = encoder.encode(prediction, content_type)
            if content_type == content_types.CSV:
                encoded_prediction = encoded_prediction.encode("utf-8")
            return encoded_prediction

    raise errors.UnsupportedFormatError(accept)